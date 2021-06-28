```@meta
EditURL = "<unknown>/src/RadialBasisFunctionModels.jl"
```

````@example RadialBasisFunctionModels
using Base: NamedTuple, promote_eltype

export RBFInterpolator
export RBFMachine, fit!, add_data!

export auto_grad, auto_jac, grad, jac, eval_and_auto_grad
export eval_and_auto_jac, eval_and_grad, eval_and_jac
````

Dependencies of this module:

````@example RadialBasisFunctionModels
using StaticPolynomials
using ThreadSafeDicts
using Memoization: @memoize
using StaticArrays
using LinearAlgebra: norm
using Lazy: @forward
using Parameters: @with_kw

for V in [:SizedVector, :MVector]
end

import Zygote as Zyg
using Zygote: Buffer
````

# Radial Basis Function Models

The module `RadialBasisFunctionModels` provides utilities to work with radial
basis function [RBF] models.
Given ``N`` data sites ``X = \{ x^1, …, x^N \} ⊂ ℝ^n`` and values
``Y = \{ y^1, …, y^N \} ⊂ ℝ``, an interpolating RBF model ``r\colon ℝ^n → ℝ``
has the form
```math
r(x) = \sum_{i=1}^N w_i φ( \| x - x^i \|_2 ) + p(x),
```
where `p` is a multivariate polynomial.
The radial function ``φ\colon [0, ∞) \to ℝ`` defines the RBF and we can solve for
the coefficients ``w`` by solving the interpolation system
```math
\begin{equation}
r( x^i ) \stackrel{!}= y^i \quad \text{for all }i=1,…,N
\label{eqn:coeff_basic}
\end{equation}
```

!!! note
    See the section about **[Getting the Coefficients](@ref)** for how we actually solve the equation system.

## Radial Basis Function Sum.

The function ``k(•) = φ(\|•\|_2)`` is radially symmetric around the origin.
``k`` is called the kernel of an RBF.

We define an abstract super type for radial functions:

````@example RadialBasisFunctionModels
abstract type RadialFunction <: Function end
````

Each Type that inherits from `RadialFunction` should implement
an evaluation method.
It takes the radius/distance ``ρ = ρ(x) = \| x - x^i \|`` from
``x`` to a specific center ``x^i``.

````@example RadialBasisFunctionModels
(φ :: RadialFunction )( ρ :: Real ) :: Real = Nothing;
nothing #hide
````

We also need the so called order of conditional positive definiteness:

````@example RadialBasisFunctionModels
cpd_order( φ :: RadialFunction) :: Int = nothing;
nothing #hide
````

The derivative can also be specified. It defaults to

````@example RadialBasisFunctionModels
df( φ :: RadialFunction, ρ ) = Zyg.gradient( φ, ρ )[1]
````

The file `radial_funcs.jl` contains various radial function implementations.
# Some Radial Functions

The **Gaussian** is defined by ``φ(ρ) = \exp \left( - (αρ)^2 \right)``, where
``α`` is a shape parameter to fine-tune the function.

````@example RadialBasisFunctionModels
"""
    Gaussian( α = 1 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = \\exp( - (α ρ)^2 ).
```
"""
@with_kw struct Gaussian{R<:Real} <: RadialFunction
    α :: R = 1
    @assert α > 0 "The shape parameter `α` must be positive."
end

function ( φ :: Gaussian )( ρ :: Real )
    exp( - (φ.α * ρ)^2 )
end

cpd_order( :: Gaussian ) = 0
df(φ :: Gaussian, ρ :: Real) = - 2 * φ.α^2 * ρ * φ( ρ )
````

The **Multiquadric** is ``φ(ρ) = - \sqrt{ 1 + (αρ)^2 }`` and also has a positive shape
parameter. We can actually generalize it to the following form:

````@example RadialBasisFunctionModels
"""
    Multiquadric( α = 1, β = 1//2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{ \\lceil β \\rceil } ( 1 + (αρ)^2 )^β
```
"""
@with_kw struct Multiquadric{R<:Real,S<:Real} <: RadialFunction
    α :: R  = 1     # shape parameter
    β :: S  = 1//2  # exponent

    @assert α > 0 "The shape parameter `α` must be positive."
    @assert β % 1 != 0 "The exponent must not be an integer."
    @assert β > 0 "The exponent must be positive."
end

function ( φ :: Multiquadric )( ρ :: Real )
    (-1)^(ceil(Int, φ.β)) * ( 1 + (φ.α * ρ)^2 )^φ.β
end

cpd_order( φ :: Multiquadric ) = ceil( Int, φ.β )
df(φ :: Multiquadric, ρ :: Real ) = (-1)^(ceil(Int, φ.β)) * 2 * φ.α * φ.β * ρ * ( 1 + (φ.α * ρ)^2 )^(φ.β - 1)
````

Related is the **Inverse Multiquadric** `` φ(ρ) = (1+(αρ)^2)^{-β}``:

````@example RadialBasisFunctionModels
"""
    InverseMultiquadric( α = 1, β = 1//2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = ( 1 + (αρ)^2 )^{-β}
```
"""
@with_kw struct InverseMultiquadric{R<:Real,S<:Real} <: RadialFunction
    α :: R  = 1
    β :: S  = 1//2

    @assert α > 0 "The shape parameter `α` must be positive."
    @assert β > 0 "The exponent must be positive."
end

function ( φ :: InverseMultiquadric )( ρ :: Real )
   ( 1 + (φ.α * ρ)^2 )^(-φ.β)
end

cpd_order( :: InverseMultiquadric ) = 0
df(φ :: InverseMultiquadric, ρ :: Real ) = - 2 * φ.α^2 * φ.β * ρ * ( 1 + (φ.α * ρ)^2 )^(-φ.β - 1)
````

The **Cubic** is ``φ(ρ) = ρ^3``.
It can also be generalized:

````@example RadialBasisFunctionModels
"""
    Cubic( β = 3 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{ \\lceil β \\rceil /2 } ρ^β
```
"""
@with_kw struct Cubic <: RadialFunction
    β :: Int = 3

    @assert β > 0 "The exponent `β` must be positive."
    @assert β % 2 != 0 "The exponent `β` must not be an even number."
end

function ( φ :: Cubic )( ρ :: Real )
    (-1)^ceil(Int, φ.β/2 ) * ρ^φ.β
end

cpd_order( φ :: Cubic ) = ceil( Int, φ.β/2 )
df(φ :: Cubic, ρ :: Real ) = (-1)^(ceil(Int, φ.β/2)) * φ.β * ρ^(φ.β - 1)
````

The thin plate spline is usually defined via
``φ(ρ) = ρ^2 \log( ρ )``.
We provide a generalized version, which defaults to
``φ(ρ) = - ρ^4 \log( ρ )``.

````@example RadialBasisFunctionModels
"""
    ThinPlateSpline( k = 2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{k+1} ρ^{2k} \\log(ρ)
```
"""
@with_kw struct ThinPlateSpline <: RadialFunction
    k :: Int = 2

    @assert k > 0 && k % 1 == 0 "The parameter `k` must be a positive integer."
end

function (φ :: ThinPlateSpline )( ρ :: T ) where T<:Real
    ρ == 0 ? zero(T) : (-1)^(φ.k+1) * ρ^(2*φ.k) * log( ρ )
end

cpd_order( φ :: ThinPlateSpline ) = φ.k + 1
df(φ :: ThinPlateSpline, ρ :: Real ) = ρ == 0 ? 0 : (-1)^(φ.k+1) * ρ^(2*φ.k - 1) * ( 2 * φ.k * log(ρ) + 1)
````

!!! note
    The thin plate spline with `k = 1` is not differentiable at `ρ=0` but we define the derivative
    as 0, which results in a continuous extension.

From an `RadialFunction` and a vector we can define a shifted kernel function.

````@example RadialBasisFunctionModels
const NumberOrVector = Union{<:Real, AbstractVector{<:Real}}

struct ShiftedKernel{RT <: RadialFunction, CT <: AbstractVector{<:Real}} <: Function
    φ :: RT
    c :: CT
end

norm2( vec ) = norm(vec, 2)

"Evaluate kernel `k` at `x - k.c`."
function (k::ShiftedKernel)( x :: AbstractVector{<:Real} )
    return k.φ( norm2( x .- k.c ) )
end
````

A vector of ``N`` kernels is a mapping ``ℝ^n → ℝ^N, \ x ↦ [ k₁(x), …, k_N(x)] ``.

````@example RadialBasisFunctionModels
_eval_vec_of_kernels( K, x ) = [k(x) for k ∈ K]

"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]`` at `x`."
( K::AbstractVector{<:ShiftedKernel})( x ) = _eval_vec_of_kernels( K, x )
````

Suppose, we have calculated the distances ``\|x - x^i\|`` beforehand.
We can save redundant effort by passing them to the radial functions of the kernels.

````@example RadialBasisFunctionModels
"Evaluate `k.φ` for distance `ρ` where `ρ` should equal `x - k.c` for the argument `x`."
eval_at_dist( k :: ShiftedKernel , ρ :: Real ) = k.φ(ρ)

"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]``, provided the distances ``[ ρ_1(x), …, ρ_{N_c}(x) ]``."
function eval_at_dist( K::AbstractVector{<:ShiftedKernel}, dists :: AbstractVector{<:Real})
    [ eval_at_dist(k,ρ) for (k,ρ) ∈ zip(K,dists) ]
end
````

Provided we have solved the interpolation system, the weights for the radial basis function
part of ``r`` are ``w``, where ``w`` is a vector of length ``N_c`` or a matrix in ``ℝ^{N_c \times k}``
where k is the number of outputs.
We treat the general case ``k\ge 1`` and always assume ``w`` to be a matrix.

````@example RadialBasisFunctionModels
struct RBFSum{
    KT <: AbstractVector{<:ShiftedKernel},
    WT <: AbstractMatrix{<:Real}
}
    kernels :: KT
    weights :: WT # can be a normal matrix or a SMatrix

    num_outputs :: Int
end
````

Make it display nicely:

````@example RadialBasisFunctionModels
function Base.show( io :: IO, rbf :: RBFSum{KT,WT} ) where {KT, WT}
    compact = get(io, :compact, false)
    if compact
        print(io, "RBFSum{$(KT), $(WT)}")
    else
        n_out, n_kernels = size(rbf.weights)
        print(io, "RBFSum\n")
        print(io, "* with $(n_kernels) kernels in an array of type $(KT)\n")
        print(io, "* and a $(n_kernels)×$(n_out) weight matrix of type $(WT).")
    end
end
````

We can easily evaluate the `ℓ`-th output of the `RBFPart`:

````@example RadialBasisFunctionModels
@doc "Evaluate outut `ℓ` of RBF sum `rbf::RBFSum`"
function (rbf :: RBFSum)(x :: AbstractVector, ℓ :: Int)
    return (rbf.weights[ℓ,:]'rbf.kernels(x))[1]
end
````

The overall output is a vector, and we also get it via matrix multiplication.

````@example RadialBasisFunctionModels
_eval_rbfsum(rbf::RBFSum, x ) = rbf.weights*rbf.kernels(x)
"Evaluate `rbf::RBFSum` at `x`."
(rbf :: RBFSum)( x :: AbstractVector ) = _eval_rbfsum(rbf, x)
````

We want to return the right type and use `_type_guard`:

````@example RadialBasisFunctionModels
_type_guard( x , :: Type{<:Vector}, :: Int ) = convert( Vector, x)
for V in [:SVector, :MVector, :SizedVector ]
    @eval _type_guard( x, ::Type{ <: $V }, n_out :: Int ) = convert($V{ n_out }, x)
end

(rbf :: RBFSum)( x :: Vector ) = _type_guard( _eval_rbfsum(rbf, x), Vector, rbf.num_outputs )
function (rbf :: RBFSum)( x :: T ) where T<:Union{SVector,MVector,SizedVector}
    return _type_guard( _eval_rbfsum(rbf, x), T, rbf.num_outputs )
end
````

As before, we allow to pass precalculated distance vectors:

````@example RadialBasisFunctionModels
function eval_at_dist( rbf::RBFSum, dists :: AbstractVector{<:Real}, ℓ :: Int )
   rbf.weights[ℓ,:]'eval_at_dist( rbf.kernels, dists )
end

function eval_at_dist( rbf :: RBFSum, dists :: AbstractVector{<:Real})
   vec(rbf.weights*eval_at_dist(rbf.kernels, dists ))
end
````

For the PolynomialTail do something similar and
use a `StaticPolynomials.PolynomialSystem` with a weight matrix.

If the polynomial degree is < 0, we use an `EmptyPolySystem`:

````@example RadialBasisFunctionModels
"Drop-In Alternative to `StaticPolynomials.PolynomialSystem` when there are no outputs."
struct EmptyPolySystem{Nvars} end
Base.length(::EmptyPolySystem) = 0
StaticPolynomials.npolynomials(::EmptyPolySystem) = 0

"Evaluate for usual vector input. (Scalar input also supported, there are no checks)"
StaticPolynomials.evaluate(:: EmptyPolySystem, :: Union{R, Vector{R}}) where R<:Real = Int[]
"Evaluate for sized input."
StaticPolynomials.evaluate(:: EmptyPolySystem{Nvars}, :: StaticVector ) where {Nvars} = SVector{0,Int}()
(p :: EmptyPolySystem)( x :: NumberOrVector) = evaluate(p, x)

function StaticPolynomials.jacobian( :: EmptyPolySystem{Nvars}, args... ) where Nvars
    Matrix{Int}(undef, 0, Nvars )
end

function StaticPolynomials.evaluate_and_jacobian( p :: EmptyPolySystem, args ... )
    return p(args...), jacobian(p, args...)
end
````

This allows for the `PolySum`. `polys` evaluates the polynomial basis and
`weights` are determined during training/fitting.

````@example RadialBasisFunctionModels
struct PolySum{
        PS <: Union{EmptyPolySystem, PolynomialSystem},
        WT <: AbstractMatrix
    }
    polys :: PS
    weights :: WT       # n_out × n_polys matrix
    num_outputs :: Int

    function PolySum( polys :: PS, weights :: WT) where{PS, WT}
        n_out, n_polys = size(weights)
        @assert npolynomials(polys) == n_polys "Number of polynomials does not macth."
        new{PS,WT}(polys, weights, n_out)
    end
end

eval_psum( p :: PolySum, x ) = p.weights * p.polys(x)
(p :: PolySum)(x :: AbstractVector ) = eval_psum( p, x )
(p :: PolySum)(x :: Vector ) = _type_guard(eval_psum(p,x), Vector, p.num_outputs )
(p :: PolySum)(x :: T) where T<:Union{SVector,MVector,SizedVector} = _type_guard( eval_psum(p,x), T, p.num_outputs)

(p :: PolySum)(x,ℓ::Int) = (p.weights[ℓ,:]'p.polys(x))[end]
````

We now have all ingredients to define the model type.

````@example RadialBasisFunctionModels
"""
    RBFModel{V}

* `V` is `true` by default. It can be set to `false` only if the number
  of outputs is 1. Then scalars are returned.

"""
struct RBFModel{V,
        RS <: RBFSum,
        PS <: PolySum }
    rbf :: RS
    psum :: PS

    # Information fields
    num_vars :: Int
    num_outputs :: Int
    num_centers :: Int
end
````

We want a model to be displayed in a sensible way:

````@example RadialBasisFunctionModels
function Base.show( io :: IO, mod :: RBFModel{V,RS,PS} ) where {V,RS,PS}
    compact = get(io, :compact, false)
    if compact
        print(io, "$(mod.num_vars)D$(mod.num_outputs)D-RBFModel{$(V)}")
    else
        print(io, "RBFModel{$(V),$(RS),$(PS)}\n")
        if V
            print(io, "\twith vector output ")
        else
            print(io, " with scalar output ")
        end
        print(io, "and $(mod.num_centers) centers, ")
        print(io, "mapping from ℝ^$(mod.num_vars) to ℝ^$(mod.num_outputs).")
    end
end
````

Evaluation is easy. We accept an additional `::Nothing` argument that does nothing
for now, but saves some typing later.

````@example RadialBasisFunctionModels
function vec_eval(mod :: RBFModel, x :: AbstractVector{<:Real}, :: Nothing)
    return mod.rbf(x) .+ mod.psum( x )
end

function scalar_eval(mod :: RBFModel, x :: AbstractVector{<:Real}, :: Nothing )
    return (mod.rbf(x) + mod.psum( x ))[1]
end

# @doc "Evaluate model `mod :: RBFModel` at vector `x`."
( mod :: RBFModel{true, RS, PS} where {RS,PS} )(x :: AbstractVector{<:Real}, ℓ :: Nothing = nothing ) = vec_eval(mod,x,ℓ)
( mod :: RBFModel{false, RS, PS} where {RS,PS} )(x :: AbstractVector{<:Real}, ℓ :: Nothing = nothing ) = scalar_eval(mod,x,ℓ)

"Evaluate scalar output `ℓ` of model `mod` at vector `x`."
function (mod :: RBFModel)( x :: AbstractVector{<:Real}, ℓ :: Int)
    return mod.rbf(x, ℓ) + mod.psum( x, ℓ )
end

# scalar input
const NothInt = Union{Nothing,Int}

function (mod :: RBFModel)(x :: Real, ℓ :: NothInt = nothing )
    @assert mod.num_vars == 1 "The model has more than 1 inputs. Provide a vector `x`, not a number."
    mod( [x,], ℓ)
end
````

## Derivatives

The easiest way to provide derivatives is via Automatic Differentiation.
We have imported `Zygote` as `Zyg`.
For automatic differentiation we need custom adjoints for some `StaticArrays`:

````@example RadialBasisFunctionModels
Zyg.@adjoint (T::Type{<:SizedMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SizedVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
````

This allows us to define the following methods:

````@example RadialBasisFunctionModels
"Return the jacobian of `rbf` at `x` (using Zygote)."
function auto_jac( rbf :: RBFModel, x :: AbstractVector{<:Real} )
    Zyg.jacobian( rbf, x )[1]
end

"Evaluate the model and return the jacobian at the same time."
function eval_and_auto_jac( rbf :: RBFModel, x :: AbstractVector{<:Real} )
    y, back = Zyg._pullback( rbf, x )

    T = eltype(y)   # TODO does this make sense?
    n = length(y)
    jac = zeros(T, n, length(x) )
    for i = 1 : length(x)
        e = [ zeros(T, i -1 ); T(1); zeros(T, n - i )  ]
        jac[i, :] .= back(e)[2]
    end

    return y, jac
end

"Return gradient of output `ℓ` of model `rbf` at point `x` (using Zygote)."
function auto_grad( rbf :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1)
    Zyg.gradient( χ -> rbf(χ, ℓ), x )[1]
end

"Evaluate output `ℓ` of the model and return the gradient."
function eval_and_auto_grad( rbf :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    y, back = Zyg._pullback( χ -> rbf(χ, ℓ)[end], x)

    grad = back( one(y) )[2]
    return y, grad
end
````

!!! note
    We need at least `ChainRules@v.0.7.64` to have `auto_grad` etc. work for StaticArrays,
    see [this issue](https://github.com/FluxML/Zygote.jl/issues/860).

But we don't need `Zygote`, because we can derive the gradients ourselves.
Assume that ``φ`` is two times continuously differentiable. \
What is the gradient of a scalar RBF model?
Using the chain rule and ``ξ = x - x^j`` we get
```math
\dfrac{∂}{∂ξ_i} \left( φ(\| ξ \|) \right)
=
φ\prime ( \| ξ \| ) \cdot
\dfrac{∂}{∂ξ_i} ( \| ξ \| )
=
φ\prime ( \| ξ \| ) \cdot
\dfrac{ξ_i}{\|ξ\|}.
```
The right term is always bounded, but not well defined for ``ξ = 0``
(see [^wild_diss] for details). \
**That is why we require ``φ'(0) \stackrel{!}= 0``.** \
We have ``\dfrac{∂}{∂x_i} ξ(x) = 1`` and thus
```math
∇r(x) = \sum_{i=1}^N \frac{w_i φ\prime( \| x - x^i \| )}{\| x - x^i \|} (x - x^i) + ∇p(x)
```

We can then implement the formula from above.
For a fixed center ``x^i`` let ``o`` be the distance vector ``x - x^i``
and let ``ρ`` be the norm ``ρ = \|o\| = \| x- x^i \|``.
Then, the gradient of a single kernel is:

````@example RadialBasisFunctionModels
function grad( k :: ShiftedKernel, o :: AbstractVector{<:Real}, ρ :: Real )
    ρ == 0 ? zero(k.c) : (df( k.φ, ρ )/ρ) .* o
end
````

In terms of `x`:

````@example RadialBasisFunctionModels
function grad( k :: ShiftedKernel, x :: AbstractVector{<:Real} )
    o = x - k.c     # offset vector
    ρ = norm2( o )  # distance
    return grad( k, o, ρ )
end
````

The jacobian of a vector of kernels follows suit:

````@example RadialBasisFunctionModels
function jacT( K :: AbstractVector{<:ShiftedKernel}, x :: AbstractVector{<:Real})
    hcat( ( grad(k,x) for k ∈ K )... )
end
# precalculated offsets and distances, 1 per kernel
function jacT( K :: AbstractVector{<:ShiftedKernel}, offsets :: AbstractVector{<:AbstractVector}, dists :: AbstractVector{<:Real} )
    hcat( ( grad(k,o,ρ) for (k,o,ρ) ∈ zip(K,offsets,dists) )... )
end
jac( K :: AbstractVector{<:ShiftedKernel}, args... ) = transpose( jacT(K, args...) )
````

Hence, the gradients of an RBFSum are easy:

````@example RadialBasisFunctionModels
function grad( rbf :: RBFSum, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    #vec( jacT( rbf.kernels, x) * rbf.weights[:,ℓ] )
    vec( rbf.weights[ℓ,:]'jac( rbf.kernels, x ) )
end

function grad( rbf :: RBFSum, offsets :: AbstractVector{<:AbstractVector}, dists :: AbstractVector{<:Real}, ℓ :: Int)
    return vec( rbf.weights[ℓ,:]'jac( rbf.kernels, offsets, dists ) )
end
````

The `grad` method looks very similar for the `PolySum`.
We obtain the jacobian of the polynomial basis system via
`PolynomialSystem.jacobian`.

````@example RadialBasisFunctionModels
function grad( psum :: PolySum, x :: AbstractVector{<:Real} , ℓ :: Int = 1)
    return vec( psum.weights[ℓ,:]'jacobian( psum.polys, x ))
end
````

For the `RBFModel` we simply combine both methods:

````@example RadialBasisFunctionModels
function _grad( mod :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    return grad(mod.rbf, x, ℓ) + grad( mod.psum, x, ℓ )
end

function grad( mod :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    return _grad(mod, x, ℓ)
end

grad( mod :: RBFModel, x :: Vector{<:Real}, ℓ :: Int = 1 ) = _type_guard( _grad(mod, x, ℓ), Vector, mod.num_vars )
function grad( mod :: RBFModel, x :: T, ℓ :: Int = 1 ) where T <: Union{SVector, MVector, SizedVector}
    return _type_guard( _grad(mod, x, ℓ), T, mod.num_vars )
end
````

We can exploit our custom evaluation methods for "distances":

````@example RadialBasisFunctionModels
function _offsets_and_dists( rbf :: RBFSum, x :: AbstractVector{<:Real} )
    offsets = [ x - k.c for k ∈ rbf.kernels ]
    dists = norm2.(offsets)
    return offsets, dists
end

function eval_and_grad( rbf :: RBFSum, offsets :: AbstractVector{<:AbstractVector}, dists :: AbstractVector{<:Real}, ℓ :: Int)
    return eval_at_dist( rbf, dists, ℓ ), grad( rbf, offsets, dists, ℓ)
end

function eval_and_grad( rbf :: RBFSum, x :: AbstractVector{<:Real}, ℓ :: Int = 1)
    offsets, dists = _offsets_and_dists(rbf, x)
    return eval_and_grad( rbf, offsets, dists, ℓ)
end
````

For the `PolySum` we use `evaluate_and_jacobian`.

````@example RadialBasisFunctionModels
function eval_and_grad( psum :: PolySum, x :: AbstractVector{<:Real}, ℓ :: Int = 1)
    res_p, J_p = evaluate_and_jacobian( psum.polys, x )
    return (psum.weights[ℓ,:]'res_p)[1], vec(psum.weights[ℓ,:]'J_p)
end
````

Combine for `RBFModel`:

````@example RadialBasisFunctionModels
function eval_and_grad( mod :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    res_rbf, g_rbf = eval_and_grad( mod.rbf, x, ℓ )
    res_polys, g_polys = eval_and_grad( mod.psum, x, ℓ )
    return res_rbf .+ res_polys, g_rbf .+ g_polys
end
````

For the jacobian, we use the same trick to save evaluations.

````@example RadialBasisFunctionModels
function jac( rbf :: RBFSum, x :: AbstractVector{<:Real} )
    offsets, dists = _offsets_and_dists(rbf, x)
    rbf.weights * jac( rbf.kernels, offsets, dists )
end
jacT(rbf :: RBFSum, args... ) = transpose( jac(rbf, args...) )

function jac( psum :: PolySum, x :: AbstractVector{<:Real} )
    psum.weights * jacobian( psum.polys, x )
end

function _jac( mod :: RBFModel, x :: AbstractVector{<:Real} )
    jac( mod.rbf, x ) + jac( mod.psum, x)
end

jac( mod :: RBFModel, x :: AbstractMatrix{<:Real} ) = _jac(mod,x)
jac( mod :: RBFModel, x :: Vector{<:Real}) = convert(Matrix, _jac(mod,x))
jac( mod :: RBFModel, x :: SVector{<:Real}) = convert( SMatrix{mod.num_outputs, mod_nmu_vars}, _jac(mod,x) )
jac( mod :: RBFModel, x :: MVector{<:Real}) = convert( MMatrix{mod.num_outputs, mod_nmu_vars}, _jac(mod,x) )
jac( mod :: RBFModel, x :: SizedVector{<:Real}) = convert( SizedMatrix{mod.num_outputs, mod_nmu_vars}, _jac(mod,x) )
````

As before, define an "evaluate-and-jacobian" function that saves evaluations:

````@example RadialBasisFunctionModels
function eval_and_jac( rbf :: RBFSum, x :: AbstractVector{<:Real} )
    offsets, dists = _offsets_and_dists(rbf, x)
    res = eval_at_dist( rbf, dists )
    J = rbf.weights * jac( rbf.kernels, offsets, dists )
    return res, J
end

function eval_and_jac( psum :: PolySum, x :: AbstractVector{<:Real} )
    res_p, J_p = evaluate_and_jacobian( psum.polys, x )
    return vec( psum.weights * res_p ), psum.weights * J_p
end

function eval_and_jac( mod :: RBFModel, x :: AbstractVector{<:Real} )
    res_rbf, J_rbf = eval_and_jac( mod.rbf, x )
    res_polys, J_polys = eval_and_jac( mod.psum, x)
    return res_rbf + res_polys, J_rbf + J_polys
end

# TODO type stable eval_and_grad and eval_and_jac ?
````

!!! note
    Hessians are not yet implemented.

For the Hessian ``Hr \colon ℝ^n \to ℝ^{n\times n}`` we need the gradients of the
component functions
```math
    ψ_j(ξ) = \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|} ξ_j
```
Suppose ``ξ ≠ 0``.
First, using the product rule, we have
```math
   \dfrac{∂}{∂ξ_i}
   \left(
   \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|} ξ_j
   \right) =
   ξ_j
   \dfrac{∂}{∂ξ_i}
   \left(
   \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|}
   \right)
   +
   \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|}
   \dfrac{∂}{∂ξ_i}
   ξ_j
```
The last term is easy because of
```math
\frac{∂}{∂ξ_i} ξ_j
=
\begin{cases}
    1 & \text{if }i = j,\\
    0 & \text{else.}
\end{cases}
```
For the first term we find
```math
   \dfrac{∂}{∂ξ_i}
   \left(
     \frac{ φ'( \left\| ξ \right\| )}
      {\|ξ\|}
   \right)
   =
   \frac{
       φ'\left(\left\| ξ \right\|\right) ∂_i \|ξ\|
       - \|ξ\| ∂_i φ'\left( \left\| ξ \right\|\right)
    }{
        \|ξ\|^2
    }
    =
    \frac{
        \dfrac{φ'(\|ξ\|)}{\|ξ\|} ξ_i - \|ξ\|φ''(\|ξ\|)\dfrac{ξ_i}{\|ξ\|}
    }{\|ξ\|^2}
```
Hence, the gradient of ``ψ_j`` is
```math
    ∇ψ_j(ξ)
    =
    \left( \frac{φ'(\|ξ\|)}{\|ξ\|^3}
    -
    \frac{φ''(\|ξ\|)}{\|ξ\|^2} \right) \cdot ξ
    -\frac{φ'(\|ξ\|)}{\|ξ\|} e^j,
```
where ``e^j ∈ ℝ^n`` is all zeros, except ``e^j_j = 1``.
For ``ξ = 0`` the first term vanishes due to L'Hôpital's rule:
```math
∇ψ_j(0) = φ''(0) e^j.
```

This file is included from within RadialBasisFunctionModels.jl #src

## Getting the Coefficients

````@example RadialBasisFunctionModels
const VecOfVecs{T} = AbstractVector{<:AbstractVector}
````

###  Polynomial Basis

Any constructor of an `RBFModel` must solve for the coefficients in ``\eqref{eqn:coeff_basic}``.
To build the equation system, we need a basis ``\{p_j\}_{1 \le j \le Q}`` of ``Π_d(ℝ^n)``.
For the interpolation system to be solvable we have to choose the
right polynomial space for ``p``.
Basically, if the RBF Kernel (or the radial function) is
*conditionally positive definite* of order ``D`` we have to
find a polynomial ``p`` with ``\deg p \ge D-1``.[^wendland]
If the kernel is CPD of order ``D=0`` we do not have to add an polynomial
and can interpolate arbitrary (distinct) data points. \

The canonical basis is ``x_1^{α_1} x_2^{α_2} … x_n^{α_n}`` with
``α_i ≥ 0`` and ``Σ_i α_i ≤ d``.
For ``\bar{d} \le d`` we can recursively get the non-negative integer solutions for
``Σ_i α_i = \bar{d}`` with the following function:

````@example RadialBasisFunctionModels
@doc """
    non_negative_solutions( d :: Int, n :: Int)

Return a matrix with columns that correspond to solution vectors
``[x_1, …, x_n]`` to the equation ``x_1 + … + x_n = d``,
where the variables are non-negative integers.
"""
function non_negative_solutions( d :: Int, n :: Int )
    if n == 1
        return fill(d,1,1)
    else
        num_sols = binomial( d + n - 1, n - 1)
        sol_matrix = Matrix{Int}(undef, n, num_sols)
        j = 1
        for i = 0 : d
            # find all solutions of length `n-1` that sum to `i`
            # if we add `d-i` to each column, then each column
            # has `n` elements and sums to `d`
            padded_shorter_solutions = vcat( d-i, non_negative_solutions(i, n-1) )
            num_shorter_sols = size( padded_shorter_solutions, 2 )
            sol_matrix[:, j : j + num_shorter_sols - 1] .= padded_shorter_solutions
            j += num_shorter_sols
        end
        return sol_matrix
    end
end
````

The polyonmial basis exponents are then given by all possible
``\bar{d}\le d``:

````@example RadialBasisFunctionModels
@doc """
    non_negative_solutions_ineq( d :: Int, n :: Int)

Return a matrix with columns that correspond to solution vectors
``[x_1, …, x_n]`` to the equation ``x_1 + … + x_n <= d``,
where the variables are non-negative integers.
"""
function non_negative_solutions_ineq( d :: Int, n :: Int )
    return hcat( (non_negative_solutions( d̄, n ) for d̄=0:d )... )
end
````

!!! note
    I did an unnecessary rewrite of `non_negative_solutions` to be
    Zygote-compatible. Therefore the matrices etc.
    `Combinatorics` has `multiexponents` which should do the same...

We **don't** use `DynamicPolynomials.jl` to generate the Polyomials **anymore**.
Zygote did overflow when there were calculations with those polynomials.
Not a problem for calculating the basis (because of we are `ignore()`ing
the basis calculation now, assuming we never want to differentiate
with respect to `n,d`),
but when constructing the outputs from them.
Instead we directly construct `StaticPolynomial`s and define a
`PolynomialSystem` that evaluates all basis polynomials.

````@example RadialBasisFunctionModels
@doc """
    canonical_basis( n:: Int, d :: Int ) :: Union{PolynomialSystem, EmptyPolySystem}

Return the canonical basis of the space of `n`-variate
polynomials of degree at most `d`.
"""
@memoize ThreadSafeDict function canonical_basis(n :: Int, d::Int, OneType :: Type = Float64)
    if d < 0
        return EmptyPolySystem{n}()
    else
        exponent_matrix = non_negative_solutions_ineq( d, n )
        one_float = OneType(1)  # `one_float` is used as coefficient(s) to guarantee floating point output
        return PolynomialSystem(
             ( Polynomial( [one_float,], e[:,:] ) for e ∈ eachcol(exponent_matrix) )...
        )
    end
end
````

### Solving the Equation System

Now let ``\{p_j\}_{1\le j\le Q}`` be a basis of the polynomial space.
Set ``P = [ p_j(x^i) ] ∈ ℝ^{N × Q}`` and ``Φ = φ(\| x^i - x^j \|)``.
In case of interpolation, the linear equation system for the
coefficients of $r$ is
```math
S c := \begin{equation}
    \begin{bmatrix}
    Φ & P \\
    P^T & 0_{Q × Q}
    \end{bmatrix}
    \begin{bmatrix}
        w \\
        λ
    \end{bmatrix}
    \stackrel{!}=
    \begin{bmatrix}
    Y
    \\
    0_Q
    \end{bmatrix}.
    \tag{I}
    \label{eqn:coeff}
\end{equation}
```

We can also use differing feature vectors and centers.
``Φ`` then becomes
``Φ = [k_j(x^i)]_{1\le i \le N_d, 1\le j \le N_c} = [φ(‖ x^i - ξ^j ‖)]``,
where we denote the number of kernel centers by ``N_c`` and the number
of feauters (``d``ata) by ``N_d``.
In the overdetermined least squares case (with pair-wise different centers and
pair-wise different features), we do away with the second row of equations in \eqref{eqn:coeff}.
The solution ``c = [w, λ]^T`` is then given by the
Moore-Penrose Pseudo-Inverse:
```math
    c = \underbrace{ ( S^T S )^{-1} S^T}_{=S^\dagger} \begin{bmatrix}
    Y
    \\
    0_Q
    \end{bmatrix}.
```
Julia automatically computes the LS solution with `S\RHS`.

!!! note
    When we have vector data ``Y ⊂ ℝ^k``, e.g. from modelling MIMO functions, then
    Julia easily allows for multiple columns in the righthand side of the interpolation
    equation system and we get weight vectors for multiple models, that can
    be thought of as one vector model ``r\colon ℝ^n \to ℝ^k``.

````@example RadialBasisFunctionModels
@doc """
    coefficients(sites, values, kernels, rad_funcs, polys )

Return the coefficient matrices `w` and `λ` for an rbf model
``r(x) = Σ_{i=1}^N wᵢ φ(\\|x - x^i\\|) + Σ_{j=1}^M λᵢ pᵢ(x)``,
where ``N`` is the length of `rad_funcs` (and `centers`) and ``M``
is the length of `polys`.

The arguments are
* an array of data sites `sites` with vector entries from ``ℝ^n``.
* an array of data values `values` with vector entries from ``ℝ^k``.
* an array of `ShiftedKernel`s.
* a `PolynomialSystem` or `EmptyPolySystem` (in case of deg = -1).
"""
function coefficients(
        sites, values, kernels, polys; mode :: Symbol = :ls
    )

    N_c = length(kernels);
    N_d = length(sites);
    Q = length(polys)

    if N_d < N_c
        error("Underdetermined models not supported yet.")
    end
    if N_d < Q
        error("Too few data sites for selectod polynomial degree. (Need at least $(Q).)")
    end

    Φ = transpose( hcat( map(kernels, sites)... ) )   # N_d × N_c
    P = transpose( hcat( map(polys, sites)... ) )       # N_d × Q
    # system matrix S and right hand side
    S = [Φ P]
    RHS = transpose( hcat(values... ) );


    return _coefficients( Φ, P, S, RHS, Val(:ls) )
end

function _coeff_matrices(coeff :: AbstractMatrix, S, RHS, N_c, Q )
    return view(coeff, 1 : N_c, :), view(coeff, N_c + 1 : N_c + Q, :), S, RHS
end

function _coeff_matrices(coeff :: StaticMatrix, S, RHS, N_c, Q )
    return coeff[ SVector{N_c}(1 : N_c), :], coeff[ SVector{Q}( N_c + 1 : N_c + Q ), :], S, RHS
end

function _coefficients( Φ, P, S, RHS, ::Val{:ls} )
    N_c = size(Φ,2); Q = size(P,2);
    coeff = S \ RHS
    return _coeff_matrices(coeff, S, RHS, N_c, Q )
end

function _coefficients( Φ, P, S, RHS, ::Val{:interpolation} )
    N_d, N_c = size(Φ); Q = size(P,2);
    @assert N_d == N_c "Interpolation requires same number of features and centers." # TODO remove assertion
    S̃ = [ S ;                               # N_d × (N_c + Q)
          P' zeros(eltype(S), Q, Q )]       # Q × N_d and Q × Q
    RHS_padded = [ RHS;
        zeros( eltype(RHS), Q, size(RHS,2) ) ];
    coeff = S̃ \ RHS_padded
    return _coeff_matrices( coeff, S̃, RHS_padded, N_c, Q )
end

 function _coefficients( Φ, P, S :: StaticMatrix, RHS :: StaticMatrix, ::Val{:interpolation} )
    N_d, N_c = size(Φ); Q = size(P,2);
    @assert N_d == N_c "Interpolation requires same number of features and centers." # TODO remove assertion

    S̃ = [S ;
         P' @SMatrix(zeros(eltype(S),Q,Q)) ];
    RHS_padded = [ RHS;
        @SMatrix(zeros(eltype(RHS), Q ,size(RHS,2)))];
    coeff = S̃ \ RHS_padded
    return _coeff_matrices( coeff, S̃, RHS_padded, N_c, Q )
end
````

We can easily impose linear equality constraints,
for example requiring interpolation only on a subset of features.
In matrix form, $I$ linear equality constraints (for ``k`` outputs) can be written
as
```math
E c = b, \quad E ∈ ℝ^{I×(N_c + Q)}, b ∈ ℝ^{I\times k},\, I,k ∈ ℕ_0.
```
Now, let $ĉ$ be the least squares solution from above.
The constrained solution is
```math
 c = ĉ - Ψ E^T ( E Ψ E^T)^{-1} ( E ĉ - b ), \; Ψ := (S^T S)^{-1}
\tag{CLS1}
\label{eqn:cls1}
```
This results from forming the Lagrangian of an equivalent minimization problem.
Let ``δ = ĉ - c ∈ ℝ^{q\times k}, q = N_c + Q,`` and define the constraint residuals
as ``γ = Eĉ - b ∈ ℝ^{I\times k}``.
The Lagrangian for minimizing ``δ^TS^TSδ`` under ``Eδ=γ`` is
```math
\begin{aligned}
    L &= δ^T S^T S δ + 2 λ^T( E δ - γ )\\
    D_δL &= 2 δ^T S^T S + 2λ^T E \\
    D_λL &= 2 δ^T E^T - 2 γ^T
\end{aligned}
```
Setting the derivatives to zero leads to \eqref{eqn:cls1} via
```math
    \begin{bmatrix}
        S^T S & E^T \\
        E & 0_{I\times I}
    \end{bmatrix}
    \begin{bmatrix}
    δ \\ λ
    \end{bmatrix}
    = \begin{bmatrix}
    0_{q\times k} \\ γ
    \end{bmatrix}
\tag{L}
\label{eqn:cls2}
```
See [^adv_eco] for details.

````@example RadialBasisFunctionModels
function constrained_coefficients(
        w :: AbstractMatrix{<:Real},
        λ :: AbstractMatrix{<:Real},
        S :: AbstractMatrix{<:Real},
        E :: AbstractMatrix{<:Real},
        b :: AbstractMatrix{<:Real}
    )
    # Using Lagrangian approach:

    ĉ = [w; λ]  # least squares solution
    γ = E*ĉ - b # constraint residuals

    I, q = size(E)
    k = size(w,2)

    A = vcat(
        [S'S E'],
        [E zeros(Int,I,I)]
    )

    RHS = [
        zeros(Int, q, k);
        γ
    ]

    δλ = A \ RHS
    δ = δλ[1 : q, :]

    c = ĉ - δ  # coefficients for constrained problem

    N_c = size(w,1)

    return c[1 : N_c, :], c[N_c+1:end, :]
end
````

For the case that mentioned above, that is, interpolation at a
subset of sites, we can easily build the ``E`` matrix from the ``S``
matrix by taking the corresponding rows.

````@example RadialBasisFunctionModels
function constrained_coefficients(
        w :: AbstractMatrix{<:Real},
        λ :: AbstractMatrix{<:Real},
        S :: AbstractMatrix{<:Real},
        RHS_ls :: AbstractMatrix{<:Real},
        interpolation_indices :: AbstractVector{Int}
    )

    E = S[interpolation_indices, :]
    b = RHS_ls[interpolation_indices, :]
    return constrained_coefficients( w, λ, S, E, b )
end
````

### The Actual, Usable Constructor

We want the user to be able to pass 1D data as scalars and use the following helpers:

````@example RadialBasisFunctionModels
ensure_vec_of_vecs( before :: AbstractVector{<:AbstractVector{<:Real}} ) = before
ensure_vec_of_vecs( before :: AbstractVector{ <:Real }) = [[x,] for x in before ]

function inner_type( vec_of_vecs :: AbstractVector{<:AbstractVector{T}}) where T
    if Base.isabstracttype(T)   # like Any if data is of mixed precision
        return Float64
    else
        return T
    end
end
````

Helpers to create kernel functions.

````@example RadialBasisFunctionModels
"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `centers`."
function make_kernels( φ_arr :: AbstractVector{<:RadialFunction}, centers :: VecOfVecs )
    @assert length(φ_arr) == length(centers)
    [ ShiftedKernel(φ, c) for (φ,c) ∈ zip( φ_arr, centers) ]
end
"Return array of `ShiftedKernel`s based function `φ` with centers from `centers`."
function make_kernels( φ :: RadialFunction, centers :: VecOfVecs )
    [ ShiftedKernel(φ, c) for c ∈ centers ]
end
````

We now have all ingredients for the basic outer constructor:

````@example RadialBasisFunctionModels
@doc """
    RBFModel( features, labels, φ = Multiquadric(), poly_deg = 1; kwargs ... )

Construct a `RBFModel` from the feature vectors in `features` and
the corresponding labels in `lables`, where `φ` is a `RadialFunction` or a vector of
`RadialFunction`s.\n
Scalar data can be used, it is transformed internally. \n
StaticArrays can be used, e.g., `features :: Vector{<:SVector}`.
Providing `SVector`s only might speed up the construction.\n
If the degree of the polynomial tail, `poly_deg`, is too small it will be set to `cpd_order(φ)-1`.

If the RBF centers do not equal the the `features`, you can use the keyword argument `centers` to
pass a list of centers. If `φ` is a vector, then the length of `centers` and `φ` must be equal and
`centers[i]` will be used in conjunction with `φ[i]` to build a `ShiftedKernel`. \n
If `features` has 1D data, the output of the model will be a 1D-vector.
If it should be a scalar instead, set the keyword argument `vector_output` to `false`.
"""
function RBFModel(
        features :: AbstractVector{ <:NumberOrVector },
        labels :: AbstractVector{ <:NumberOrVector },
        φ :: Union{RadialFunction,AbstractVector{<:RadialFunction}} = Multiquadric(),
        poly_deg :: Int = 1;
        centers :: AbstractVector{ <:NumberOrVector } = Vector{Float16}[],
        interpolation_indices :: AbstractVector{ <: Int } = Int[],
        vector_output :: Bool = true,
        coeff_mode :: Symbol = :auto
    )

    # Basic Data integrity checks
    @assert !isempty(features) "Provide at least 1 feature vector."
    @assert !isempty(labels) "Provide at least 1 label vector."
    num_vars = length(features[1])
    num_outputs = length(labels[1])
    @assert all( length(s) == num_vars for s ∈ features ) "All features must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ labels ) "All labels must have same dimension."

    num_sites = length(features)
    num_vals = length(labels)
    @assert num_sites == num_vals "Provide as many features as labels."

    sites = ensure_vec_of_vecs(features)
    values = ensure_vec_of_vecs(labels)
    if !isempty(centers)
        @assert all( length(c) == num_vars for c ∈ centers ) "All centers must have dimension $(num_vars)."
        C = ensure_vec_of_vecs(centers)
    else
        C = copy(sites)
    end
    num_centers = length(C)

    kernels = make_kernels(φ, C)

    poly_precision = promote_type(Float16, inner_type(sites))
    poly_basis_sys = Zyg.ignore() do
        canonical_basis( num_vars, poly_deg, poly_precision )
    end

    if coeff_mode == :auto
        can_interpolate_uniquely = φ isa RadialFunction ? poly_deg >= cpd_order(φ) - 1 : all( poly_deg >= cpd_order(phi) - 1 for phi in φ )
        coeff_mode = num_sites == num_centers && can_interpolate_uniquely ? :interpolation : :ls
    end

    w, λ, S, RHS = coefficients( sites, values, kernels, poly_basis_sys; mode = coeff_mode )

    if !isempty(interpolation_indices)
        w, λ = constrained_coefficients( w, λ, S, RHS, interpolation_indices)
    end

    # build output polynomials
    poly_sum = PolySum( poly_basis_sys, transpose(λ) )

    # build RBF system
    rbf_sys = RBFSum(kernels, transpose(w), num_outputs)

    # vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : true

    return RBFModel{vec_output, typeof(rbf_sys), typeof(poly_sum)}(
         rbf_sys, poly_sum, num_vars, num_outputs, num_centers
    )
end
````

### Special Constructors

We offer some specialized models (that simply wrap the main type).

````@example RadialBasisFunctionModels
struct RBFInterpolationModel
    model :: RBFModel
end
(mod :: RBFInterpolationModel)(args...) = mod.model(args...)
@forward RBFInterpolationModel.model grad, jac, jacT, auto_grad, auto_jac
````

The constructor is a tiny bit simpler and additional checks take place:

````@example RadialBasisFunctionModels
"""
    RBFInterpolationModel(features, labels, φ, poly_deg; kwargs… )

Build a model interpolating the feature-label pairs.
Does not accept `center` keyword argument.
"""
function RBFInterpolationModel(
        features :: AbstractVector{ <:NumberOrVector },
        labels :: AbstractVector{ <:NumberOrVector },
        φ :: Union{RadialFunction,AbstractVector{<:RadialFunction}} = Multiquadric(),
        poly_deg :: Int = 1;
        vector_output :: Bool = true,
    )
    @assert length(features) == length(labels) "Provide as many features as labels!"

    if poly_deg < cpd_order(φ) - 1
        @warn "Polyonmial degree too small for interpolation. Using $(cpd_order(φ)-1)."
        poly_deg = max( poly_deg,  cpd_order(φ) - 1 )
    end

    mod = RBFModel(features, labels, φ, poly_deg; vector_output, coeff_mode = :interpolation)
    return RBFInterpolationModel( mod )
end
````

We want to provide a convenient alternative constructor for interpolation models
so that the radial function can be defined by passing a `Symbol` or `String`.

````@example RadialBasisFunctionModels
const SymbolToRadialConstructor = NamedTuple((
    :gaussian => Gaussian,
    :multiquadric => Multiquadric,
    :inv_multiquadric => InverseMultiquadric,
    :cubic => Cubic,
    :thin_plate_spline => ThinPlateSpline
))

"Obtain a `RadialFunction` from its name and constructor arguments."
function _get_rad_func( φ_symb :: Union{Symbol, String}, φ_args )

    # which radial function to use?
    radial_symb = Symbol( lowercase( string( φ_symb ) ) )
    if !(radial_symb ∈ keys(SymbolToRadialConstructor))
        @warn "Radial Funtion $(radial_symb) not known, using Gaussian."
        radial_symb = :gaussian
    end

    constructor = SymbolToRadialConstructor[radial_symb]
    if isnothing(φ_args)
        φ = constructor()
    else
        φ = constructor( φ_args... )
    end

    return φ
end
````

The alternative constructors are build programmatically:

````@example RadialBasisFunctionModels
for op ∈ [ :RBFInterpolationModel, :RBFModel ]
    @eval begin
        function $op(
                features :: AbstractVector{ <:NumberOrVector },
                labels :: AbstractVector{ <:NumberOrVector },
                φ_symb :: Union{Symbol, String},
                φ_args = nothing,
                poly_deg :: Int = 1; kwargs...
            )

            φ = _get_rad_func( φ_symb, φ_args )
            return $op(features, labels, φ, poly_deg; kwargs... )
        end
    end
end
````

### Container with Training Data

The RBF Machine is similar in design to what an MLJ machine does:
Training data (feature and label **vectors**) are stored and can be added.
The inner model is trained with `fit!`.

**TODO** In the future, we can customize the `fit!` method when updating a model
to only consider *new* training data.
This also makes type conversion of the whole data arrays unnecessary.

````@example RadialBasisFunctionModels
"""
    RBFMachine(; features = Vector{Float64}[], labels = Vector{Float64}[],
    kernel_name = :gaussian, kernel_args = nothing, poly_deg = 1)

A container holding an inner `model :: RBFModel` (or `model == nothing`).
An array of arrays of features is stored in the `features` field.
Likewise for `labels`.
The model is trained with `fit!` and can then be evaluated.
"""
@with_kw mutable struct RBFMachine{
        FT <: AbstractVector{<:AbstractVector{<:AbstractFloat}},
        LT <: AbstractVector{<:AbstractVector{<:AbstractFloat}},
    }
    features :: FT = Vector{Float64}[]
    labels :: LT = Vector{Float64}[]
    kernel_name :: Symbol = :gaussian
    kernel_args :: Union{Nothing, Vector{Float64}} = nothing
    poly_deg :: Int = 1

    model :: Union{Nothing,RBFModel} = nothing
    valid :: Bool = false   # is model trained on all data sites?

    @assert let T = eltype( Base.promote_eltype(FT, LT) ),
        K = isnothing(kernel_args) ? nothing : T.(kernel_args),
        φ = _get_rad_func( kernel_name, K );
        poly_deg >= cpd_order(φ) - 1
    end "Polynomial degree too low for interpolation."
end

"Return floating point type of training data elements."
_precision( :: RBFMachine{FT,LT} ) where {FT,LT} = eltype( Base.promote_eltype(FT, LT) )

"Return kernel arguments converted to minimum required precision."
function _kernel_args( mach :: RBFMachine )
    if isnothing( mach.kernel_args )
        return mach.kernel_args
    else
        T = promote_type( Float16, _precision(mach) )
        return T.(mach.kernel_args)
    end
end

"Fit `mach :: RBFMachine` to the training data."
function fit!( mach :: RBFMachine )::Nothing
    @assert length(mach.features) > 0 "Provide at least one data sample."
    num_needed =  binomial( mach.poly_deg + length(mach.features[1]), mach.poly_deg)
    @assert length(mach.features) >= num_needed "Too few data sites for selected polynomial degree (need $(num_needed))."

    inner_model = RBFModel(
        mach.features,
        mach.labels,
        mach.kernel_name,
        _kernel_args(mach),
        mach.poly_deg
    )
    mach.model = inner_model
    mach.valid = true
    return nothing
end
````

Forward evaluation methods of inner model:

````@example RadialBasisFunctionModels
( mach :: RBFMachine )(args...) = mach.model(args...)
@forward RBFMachine.model grad, jac, jacT, auto_grad, auto_jac
````

Methods to add features and labels:

````@example RadialBasisFunctionModels
"Add a feature vector(s) and a label(s) to the `machine` container."
function add_data!(
        m :: RBFMachine, features :: AbstractVector{<:AbstractVector}, labels :: AbstractVector{<:AbstractVector}
    ) :: Nothing
    @assert length(features) == length(labels) "Provide same number of features and labels."
    @assert all( length(f) == length(features[1]) for f in features ) "Features must have same length."
    @assert all( length(l) == length(labels[1]) for l in labels ) "Labels must have same length"
    @assert isempty(m.features) || length(m.features[1]) == length(features[1]) && length(m.labels[1]) == length(labels[1]) "Length doesnt match previous data."
    append!(m.features, features)
    append!(m.labels, labels)
    m.valid = false
    return nothing
end

function add_data!(
        m :: RBFMachine, feature :: AbstractVector{<:AbstractFloat}, label:: AbstractVector{<:AbstractFloat}
    ) :: Nothing
    return add_data!(m, [ feature, ], [label, ])
end
````

Convenience methods to "reset" a machine:

````@example RadialBasisFunctionModels
function Base.empty!( m :: RBFMachine ) :: Nothing
    empty!(m.features)
    empty!(m.labels)
    m.model = nothing
    m.valid = false
    return nothing
end

function Base.isempty(m :: RBFMachine ) :: Bool
    isempty( m.features ) && isempty( m.labels ) && isnothing(m.model)
end
````

````@example RadialBasisFunctionModels
include("mlj_interface.jl")
````

[^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
[^wendland]: “Scattered Data Approximation”, Wendland
[^adv_eco]: “Advanced Econometrics“, Takeshi Amemiya

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

