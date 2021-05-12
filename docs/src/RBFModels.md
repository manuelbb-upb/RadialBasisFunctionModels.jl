```@meta
EditURL = "<unknown>/src/RBFModels.jl"
```

````@example RBFModels
export auto_grad, auto_jac, grad, jac
````

Dependencies of this module:

````@example RBFModels
import DynamicPolynomials as DP
using StaticPolynomials
using ThreadSafeDicts
using Memoize: @memoize
using StaticArrays
using LinearAlgebra: norm
using Lazy: @forward

import Flux.Zygote as Zyg
#using Flux.Zygote: Buffer, @adjoint
````

# Radial Basis Function Models

The sub-module `RBFModels` provides utilities to work with radial
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
r( x^i ) \stackrel{!}= y^i \quad \text{for all }i=1,…,N
```

For the interpolation system to be solvable we have to choose the
right polynomial space for ``p``.
Basically, if the RBF Kernel (or the radial function) is
*conditionally positive definite* of order ``D`` we have to
find a polynomial ``p`` with ``\deg p \ge D-1``.[^wendland]
If the kernel is CPD of order ``D=0`` we do not have to add an polynomial
and can interpolate arbitrary (distinct) data points. \
Now let ``\{p_j}_{1\le j\le Q}`` be a basis of the polynomial space.
Set ``P = [ p_j(x^i) ] ∈ ℝ^{N × Q}`` and ``Φ = φ(\| x^i - x^j \|)``.
In case of interpolation, the linear equation system for the coefficients of $r$ is
```math
    \begin{bmatrix}
    Φ & P \\
    P^T & 0_{Q × Q}
    \end{bmatrix}
    \begin{bmatrix}
        w \\
        λ
    \end{bmatrix}
    =
    \begin{bmatrix}
    Y
    \\
    0_Q
    \end{bmatrix}.
```
We can also use differing feature vectors and centers. It is also possible to
determine a least squarse solution to a overdetermined system.
Hence, we will denote the number of kernel centers by ``N_c`` from now on.

!!! note
    When we have vector data ``Y ⊂ ℝ^k``, e.g. from modelling MIMO functions, then
    Julia easily allows for multiple columns in the righthand side of the interpolation
    equation system and we get weight vectors for multiple models, that can
    be thought of as one vector models ``r\colon ℝ^n \to ℝ``.

!!! note
    See the section about **Constructors** for how we actually solve the equation system.

## Radial Basis Function Sum.

The function ``k(•) = φ(\|•\|_2)`` is radially symmetric around the origin.
``k`` is called the kernel of an RBF.

We define an abstract super type for radial functions:

````@example RBFModels
abstract type RadialFunction <: Function end
````

Each Type that inherits from `RadialFunction` should implement
an evaluation method.
It takes the radius/distance ``ρ = ρ(x) = \| x - x^i \|`` from
``x`` to a specific center ``x^i``.

````@example RBFModels
(φ :: RadialFunction )( ρ :: Real ) :: Real = Nothing;
nothing #hide
````

We also need the so called order of conditional positive definiteness:

````@example RBFModels
cpd_order( φ :: RadialFunction) :: Int = nothing;
nothing #hide
````

The derivative can also be specified. It defaults to

````@example RBFModels
df( φ :: RadialFunction, ρ ) = Zyg.gradient( φ, ρ )[1]
````

The file `radial_funcs.jl` contains various radial function implementations.
# Some Radial Functions

The **Gaussian** is defined by ``φ(ρ) = \exp \left( - (αρ)^2 \right)``, where
``α`` is a shape parameter to fine-tune the function.

````@example RBFModels
"""
    Gaussian( α = 1 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = \\exp( - (α ρ)^2 ).
```
"""
struct Gaussian <: RadialFunction
    α :: Real

    Gaussian( α :: Real = 1 ) = begin
        @assert α > 0 "The shape parameter `α` must be positive."
        return new(α)
    end
end

function ( φ :: Gaussian )( ρ :: Real )
    exp( - (φ.α * ρ)^2 )
end

cpd_order( :: Gaussian ) = 0
df(φ :: Gaussian, ρ :: Real) = - 2 * φ.α * φ( ρ )
````

The **Multiquadric** is ``φ(ρ) = - \sqrt{ 1 + (αρ)^2 }`` and also has a positive shape
parameter. We can actually generalize it to the following form:

````@example RBFModels
"""
    Multiquadric( α = 1, β = 1//2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{ \\lceil β \\rceil } ( 1 + (αρ)^2 )^β
```
"""
struct Multiquadric <: RadialFunction
    α :: Real   # shape parameter
    β :: Real   # exponent

    Multiquadric(α = 1, β = 1//2 ) = begin
        @assert α > 0 "The shape parameter `α` must be positive."
        @assert β % 1 != 0 "The exponent must not be an integer."
        @assert β > 0 "The exponent must be positive."
        new(α,β)
    end
end

function ( φ :: Multiquadric )( ρ :: Real )
    (-1)^(ceil(Int, φ.β)) * ( 1 + (φ.α * ρ)^2 )^φ.β
end

cpd_order( φ :: Multiquadric ) = ceil( Int, φ.β )
df(φ :: Multiquadric, ρ :: Real ) = (-1)^(ceil(Int, φ.β)) * 2 * φ.α * φ.β * ρ * ( 1 + (φ.α * ρ)^2 )^(φ.β - 1)
````

Related is the **Inverse Multiquadric** `` φ(ρ) = (1+(αρ)^2)^{-β}`` is related:

````@example RBFModels
"""
    InverseMultiquadric( α = 1, β = 1//2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = ( 1 + (αρ)^2 )^{-β}
```
"""
struct InverseMultiquadric <: RadialFunction
    α :: Real
    β :: Real

    InverseMultiquadric( α = 1, β = 1//2 ) = begin
        @assert α > 0 "The shape parameter `α` must be positive."
        @assert β > 0 "The exponent must be positive."
        new(α, β)
    end
end

function ( φ :: InverseMultiquadric )( ρ :: Real )
   ( 1 + (φ.α * ρ)^2 )^(-φ.β)
end

cpd_order( :: InverseMultiquadric ) = 0
df(φ :: InverseMultiquadric, ρ :: Real ) = - 2 * φ.α^2 * φ.β * ρ * ( 1 + (φ.α * ρ)^2 )^(-φ.β - 1)
````

The **Cubic** is ``φ(ρ) = ρ^3``.
It can also be generalized:

````@example RBFModels
"""
    Cubic( β = 3 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{ \\lceil β \\rceil /2 } ρ^β
```
"""
struct Cubic <: RadialFunction
    β :: Real

    Cubic( β :: Real = 3 ) = begin
        @assert β > 0 "The exponent `β` must be positive."
        @assert β % 2 != 0 "The exponent `β` must not be an even number."
        new(β)
    end
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

````@example RBFModels
"""
    ThinPlateSpline( k = 2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{k+1} ρ^{2k} \\log(ρ)
```
"""
struct ThinPlateSpline <: RadialFunction
    k :: Int

    ThinPlateSpline( k :: Real = 2 ) = begin
        @assert k > 0 && k % 1 == 0 "The parameter `k` must be a positive integer."
        new( Int(k) )
    end
end

function (φ :: ThinPlateSpline )( ρ :: Real )
    (-1)^(k+1) * ρ^(2*k) * log( ρ )
end

cpd_order( φ :: ThinPlateSpline ) = φ.k + 1
df(φ :: ThinPlateSpline, ρ :: Real ) = ρ == 0 ? 0 : (-1)^(φ.k+1) * ρ^(2*φ.k - 1) * ( 2 * φ.k * log(ρ) + 1)
````

From an `RadialFunction` and a vector we can define a shifted kernel function.
We allow evaluation for statically sized vectors, too:

````@example RBFModels
const StatVec{T} = Union{SVector{I,T}, SizedVector{I,T,V}} where {I,V}
const AnyVec{T} = Union{Vector{T}, StatVec{T}}
const AnyMat = Union{Matrix, SMatrix, SizedMatrix}

struct ShiftedKernel <: Function
    φ :: RadialFunction
    c :: AnyVec
end

norm2( vec ) = norm(vec, 2)

"Evaluate kernel `k` at `x - k.c`."
function (k::ShiftedKernel)( x :: AnyVec{<:Real} )
    return k.φ( norm2( x - k.c ) )
end
````

A vector of ``N`` kernels is a mapping ``ℝ^n → ℝ^N, \ x ↦ [ k₁(x), …, k_N(x)] ``.

````@example RBFModels
"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]`` at `x`."
function ( K::AnyVec{ShiftedKernel})( x :: AnyVec{<:Real} )
    [ k(x) for k ∈ K ]
end
````

Suppose, we have calculated the distances ``\|x - x^i\|`` beforehand.
We can save redundant effort by passing them to the radial fucntions of the kernels.

````@example RBFModels
"Evaluate `k.φ` for distance `ρ` where `ρ` should equal `x - k.c` for the argument `x`."
eval_at_dist( k :: ShiftedKernel , ρ :: Real ) = k.φ(ρ)

"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]``, provided the distances ``[ ρ_1(x), …, ρ_{N_c}(x) ]``."
function eval_at_dist( K::AnyVec{ShiftedKernel}, dists :: Vector{<:Real})
    [ eval_at_dist(k,ρ) for (k,ρ) ∈ zip(K,dists) ]
end
````

Provided we have solved the interpolation system, the weights for the radial basis function
part of ``r`` are ``w``, where ``w`` is a vector of length ``N_c`` or a matrix in ``ℝ^{N_c \times k}``
where k is the number of outputs.
We treat the general case ``k\ge 1`` and always assume ``w`` to be a matrix.

````@example RBFModels
struct RBFSum
    kernels :: AnyVec{ShiftedKernel}
    weights :: AnyMat # weigth vectors, one for each output

    # information fields
    num_vars :: Int
    num_centers :: Int
    num_outputs :: Int

end
````

We can easily evaluate the `ℓ`-th output of the `RBFPart`.

````@example RBFModels
"Evaluate output `ℓ` of RBF sum `rbf::RBFSum`"
function (rbf :: RBFSum)(x :: AnyVec{<:Real}, ℓ :: Int)
    (rbf.kernels(x)'rbf.weights[:,ℓ])[1]
end
````

Use the above method for vector-valued evaluation of the whole sum:

````@example RBFModels
"Evaluate `rbf::RBFSum` at `x`."
(rbf::RBFSum)( x :: AnyVec{<:Real} ) = vec(rbf.kernels(x)'rbf.weights)
````

As before, we allow to pass precalculated distance vectors:

````@example RBFModels
function eval_at_dist( rbf::RBFSum, dists :: AnyVec{<:Real}, ℓ :: Int )
   eval_at_dist( rbf.kernels, dists )'rbf.weights
end

function eval_at_dist( rbf :: RBFSum, dists :: AnyVec{<:Real})
   vec(eval_at_dist(rbf.kernels, dists )'rbf.weights)
end
````

For the PolynomialTail we use a `StaticPolynomials.PolynomialSystem`. \
We now have all ingredients to define the model type.

````@example RBFModels
"""
    RBFModel{V}

* `V` is `true` by default. It can be set to `false` only if the number
  of outputs is 1. Then scalars are returned.

"""
struct RBFModel{V}
    rbf :: RBFSum
    polys :: PolynomialSystem

    # Information fields
    num_vars :: Int
    num_outputs :: Int
    num_centers :: Int
    #=
    function RBFModel( rbf :: RBFSum, polys :: PolynomialSystem, V :: Bool )
        num_centers = rbf.num_centers
        num_vars = num_centers > 0 ? length(rbf.kernels[1].c) : nvariables(polys)
        num_outputs = rbf.num_outputs > 0 ? rbf.num_outputs : npolynomials(polys)

        vec_out = num_outputs == 1 ? V : true
        return new{vec_out}(
            num_vars,
            num_outputs,
            num_centers
        )
    end
    =#
end
````

We want a model to be displayed in a sensible way:

````@example RBFModels
function Base.show( io :: IO, mod :: RBFModel{V} ) where V
    compact = get(io, :compact, false)
    if compact
        print(io, "$(mod.num_vars)D$(mod.num_outputs)D-RBFModel{$(V)}")
    else
        print(io, "RBFModel\n")
        if !V print(io, "* with scalar output\n") end
        print(io, "* with $(mod.num_centers) centers\n")
        print(io, "* mapping from ℝ^$(mod.num_vars) to ℝ^$(mod.num_outputs).")
    end
end
````

Evaluation is easy. We accept an additional `::Nothing` argument that does nothing
for now, but saves some typing later.

````@example RBFModels
function vec_eval(mod :: RBFModel, x :: AnyVec{<:Real}, :: Nothing)
    return mod.rbf(x) .+ mod.polys( x )
end

function scalar_eval(mod :: RBFModel, x :: AnyVec{<:Real}, :: Nothing )
    return (mod.rbf(x) .+ mod.polys( x ))[1]
end

"Evaluate model `mod :: RBFModel` at vector `x`."
( mod :: RBFModel{true} )(x :: AnyVec{<:Real}, ℓ :: Nothing = nothing ) = vec_eval(mod,x,ℓ)
( mod :: RBFModel{false} )(x :: AnyVec{<:Real}, ℓ :: Nothing = nothing ) = scalar_eval(mod,x,ℓ)

"Evaluate scalar output `ℓ` of model `mod` at vector `x`."
function (mod :: RBFModel)( x :: AnyVec{<:Real}, ℓ :: Int)
    return mod.rbf(x, ℓ) .+ mod.polys.polys[ℓ]( x )
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
We have imported `Flux.Zygote` as `Zyg`.
For automatic differentiation we need custom adjoints for some `StaticArrays`:

````@example RBFModels
Zyg.@adjoint (T::Type{<:SizedMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SizedVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
````

This allows us to define the following methods:

````@example RBFModels
"Return the jacobian of `rbf` at `x` (using Zygote)."
function auto_jac( rbf :: RBFModel, x :: AnyVec{<:Real} )
    Zyg.jacobian( rbf, x )[1]
end

"Evaluate the model and return the jacobian at the same time."
function eval_and_auto_jac( rbf :: RBFModel, x :: AnyVec{<:Real} )
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
function auto_grad( rbf :: RBFModel, x :: AnyVec{<:Real}, ℓ :: Int = 1)
    Zyg.gradient( χ -> rbf(χ, ℓ), x )[1]
end

"Evaluate output `ℓ` of the model and return the gradient."
function eval_and_auto_grad( rbf :: RBFModel, x :: AnyVec{<:Real}, ℓ :: Int = 1 )
    y, back = Zyg._pullback( χ -> rbf(χ, ℓ)[end], x)

    grad = back( one(y) )[2]
    return y, grad
end
````

!!! note
    We need at least `ChainRules@v.0.7.64` to have `auto_grad` etc. work for StaticArrays,
    see [this issue](https://github.com/FluxML/Zygote.jl/issues/860).

!!! note
    The above methods do not work if x is a StaticArray due to StaticPolynomials not knowing the
    custom adjoints. Maybe extendings `ChainRulesCore` helps?

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

````@example RBFModels
function grad( k :: ShiftedKernel, o :: AnyVec{<:Real}, ρ :: Real )
    ρ == 0 ? zero(k.c) : (df( k.φ, ρ )/ρ) .* o
end
````

In terms of `x`:

````@example RBFModels
function grad( k :: ShiftedKernel, x :: AnyVec{<:Real} )
    o = x - k.c     # offset vector
    ρ = norm2( o )  # distance
    return grad( k, o, ρ )
end
````

The jacobion of a vector of kernels follows suit:

````@example RBFModels
function jacT( K :: AnyVec{ShiftedKernel}, x :: AnyVec{<:Real})
    hcat( ( grad(k,x) for k ∈ K )... )
end
# precalculated offsets and distances, 1 per kernel
function jacT( K :: AnyVec{ShiftedKernel}, offsets :: AnyVec{<:AnyVec}, dists :: AnyVec{<:Real} )
    hcat( ( grad(k,o,ρ) for (k,o,ρ) ∈ zip(K,offsets,dists) )... )
end
jac( K :: AnyVec{ShiftedKernel}, args... ) = transpose( jacT(K, args...) )
````

Hence, the gradients of an RBFSum are easy:

````@example RBFModels
function grad( rbf :: RBFSum, x :: AnyVec{<:Real}, ℓ :: Int = 1 )
    vec( jacT( rbf.kernels, x) * rbf.weights[:,ℓ] )
end

function grad( rbf :: RBFSum, offsets :: AnyVec{<:AnyVec}, dists :: AnyVec{<:Real}, ℓ :: Int)
    return vec( jacT( rbf.kernels, offsets, dists ) * rbf.weights[:,ℓ] )
end

function grad( mod :: RBFModel, x :: AnyVec{<:Real}, ℓ :: Int = 1 )
    grad(mod.rbf, x, ℓ) + gradient( mod.polys.polys[ℓ], x )
end
````

We can exploit our custom evaluation methods for "distances":

````@example RBFModels
function eval_and_grad( rbf :: RBFSum, offsets :: AnyVec{<:AnyVec}, dists :: AnyVec{<:Real}, ℓ :: Int)
    return eval_at_dist( rbf, dists, ℓ ), grad( rbf, offsets, dists, ℓ)
end

function eval_and_grad( rbf :: RBFSum, x :: AnyVec{<:Real}, ℓ :: Int = 1)
    offsets = [ x - k.c for k ∈ rbf.kernels ]
    dists = norm2.(offsets)
    return eval_and_grad( rbf, offsets, dists, ℓ)
end
````

For the jacobian, we use this trick to save evaluations, too.

````@example RBFModels
function jacT( rbf :: RBFSum, x :: AnyVec{<:Real} )
    offsets = [ x - k.c for k ∈ rbf.kernels ]
    dists = norm2.(offsets)
    jacT( rbf.kernels, offsets, dists )*rbf.weights
end
jac(rbf :: RBFSum, args... ) = transpose( jacT(rbf, args...) )

function jac( mod :: RBFModel, x :: AnyVec{<:Real} )
    jac( mod.rbf, x) + jacobian( mod.polys, x )
end
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

This file is included from within RBFModels.jl #src

## Constructors
###  Polynomial Basis

Any constructor of an `RBFModel` must solve for the coefficients.
To build the equation system, we need a basis ``\{p_j\}_{1 \le j \le Q}`` of ``Π_d(ℝ^n)``.

The canonical basis is ``x_1^{α_1} x_2^{α_2} … x_n^{α_n}`` with
``α_i ≥ 0`` and ``Σ_i α_i ≤ d``.
For ``\bar{d} \le d`` we can recursively get the non-negative integer solutions for
``Σ_i α_i = \bar{d}`` with the following function:

````@example RBFModels
@doc """
    non_negative_solutions( d :: Int, n :: Int)

Return array of solution vectors ``[x_1, …, x_n]`` to the equation
``x_1 + … + x_n = d``
where the variables are non-negative integers.
"""
function non_negative_solutions( d :: Int, n :: Int )
    if n == 1
        return d
    else
        solutions = [];
        for i = 0 : d
            # make RHS smaller by and find all solutions of length `n-1`
            # then concatenate with difference `d-i`
            for shorter_solution ∈ non_negative_solutions( i, n - 1)
                push!( solutions, [ d-i ; shorter_solution ] )
            end
        end
        return solutions
    end
end
````

We use `DynamicPolynomials.jl` to generate the Polyomials.
Furthermore, we employ Memoization (via `Memoize.jl` and `ThreadSafeDicts`)
to save the result for successive usage.

````@example RBFModels
@doc """
    canonical_basis( n:: Int, d :: Int )

Return the canonical basis of the space of `n`-variate
polynomials of degree at most `d`.
"""
@memoize ThreadSafeDict function canonical_basis( n :: Int, d :: Int )
    DP.@polyvar Xvar[1 : n]
    basis = DP.Polynomial{true,Int}[] # list of basis polynomials
    for d̄ = 0 : d
        for multi_exponent ∈ non_negative_solutions( d̄, n )
            push!( basis, DP.Polynomial(prod( Xvar .^ multi_exponent ) ))
        end
    end
    return basis
end
````

### Solving the Equation System
For now, we use the `\` operator to solve `A * coeff = RHS`.
Furthermore, we allow for different interpolation `sites` and
RBF centers by allowing for passing `kernels`.

````@example RBFModels
const VecOfVecs{T} = AnyVec{<:AnyVec{<:T}}

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
* an array `polys` of polynomial basis functions.
"""
function coefficients(
        sites :: VecOfVecs,
        values :: VecOfVecs,
        kernels :: AnyVec{ShiftedKernel},
        polys :: Vector{<:DP.Polynomial};
    )

    n_out = length(values[1])

    # Φ-matrix, N columns =̂ basis funcs, rows =̂ sites
    N = length(kernels);
    Φ = hcat( (k.(sites) for k ∈ kernels)... )

    # P-matrix, N × Q
    Q = length(polys)
    P = hcat( (p.(sites) for p ∈ polys)... )

    # system matrix A
    Z = zeros( eltype(Φ), Q, Q )
    A = [ Φ  P;
          P' Z ];

    # build rhs
    RHS = [
        transpose( hcat( values... ) );
        zeros( eltype(eltype(values)), Q, n_out )
    ];

    # solve system
    coeff = A \ RHS

    # return w and λ
    return coeff[1 : N, :], coeff[N+1 : end, :]
end
````

### The Actual, Usable Constructor

We want the user to be able to pass 1D data as scalars and use the following helpers:

````@example RBFModels
const NumberOrVector = Union{<:Real, AnyVec{<:Real}}

function ensure_vec_of_vecs( before :: AnyVec{ <:Real } )
    [ SVector{1}([vec,]) for vec ∈ before ]
end
ensure_vec_of_vecs( before :: AnyVec{ <:AnyVec } ) = before
````

Helpers to create kernel functions. Should return `SVector` when appropriate.

````@example RBFModels
"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `centers`."
function make_kernels( φ_arr :: AnyVec{<:RadialFunction}, centers :: VecOfVecs )
    @assert length(φ_arr) == length(centers)
    [ ShiftedKernel(φ_arr[i], centers[i]) for i = eachindex( centers ) ]
end
"Return array of `ShiftedKernel`s based function `φ` with centers from `centers`."
function make_kernels( φ :: RadialFunction, centers :: VecOfVecs )
    [ ShiftedKernel(φ, centers[i]) for i = eachindex( centers ) ]
end
````

We use these methods to construct the RBFSum of a model

````@example RBFModels
function RBFSum( kernels :: AnyVec{<:ShiftedKernel}, weights :: AnyMat,
        num_vars :: Int = -1, num_centers ::Int = -1, num_outputs :: Int = -1;
        static_arrays :: Bool = true
    )
    if num_vars < 0 num_vars = length(kernels[1].c) end
    if num_centers < 0 num_centers = length(kernels) end
    if num_outputs < 0 num_outputs = length(weights) end

    # Sized Matrix?
    @assert size(weights) == (num_centers, num_outputs) "Weights must have dimensions $((num_centers, num_outputs)) instead of $(size(weights))."
    if num_centers * num_outputs < 100 && static_arrays
        if !( weights isa StaticArray)
            weights = SMatrix{num_centers, num_outputs}(weights)
        end
    end

    #w_vecs = copy.(eachcol(weights))

    RBFSum( kernels, weights, num_vars, num_centers, num_outputs)
end
````

We now have all ingredients for the basic outer constructor:

````@example RBFModels
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
        features :: AnyVec{ <:NumberOrVector },
        labels :: AnyVec{ <:NumberOrVector },
        φ :: Union{RadialFunction,AnyVec{<:RadialFunction}} = Multiquadric(),
        poly_deg :: Int = 1;
        centers :: AnyVec{ <:NumberOrVector } = Vector{Float16}[],
        interpolation_indices :: AnyVec{ <: Int } = Int[],
        vector_output :: Bool = true,
        static_arrays :: Bool = true
    )

    # Basic Data integrity checks
    @assert !isempty(features) "Provide at least 1 feature vector."
    @assert !isempty(labels) "Provide at least 1 label vector."
    num_vars = length(features[1])
    num_outputs = length(labels[1])
    @assert all( length(s) == num_vars for s ∈ features ) "All features must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ labels ) "All labels must have same dimension."

    if !isempty( centers )
        @assert all( length(s) == num_vars for s ∈ centers ) "All centers must have dimension $(num_vars)."
    else
        centers = features
    end

    sites = ensure_vec_of_vecs(features)
    values = ensure_vec_of_vecs(labels)
    centers = ensure_vec_of_vecs(centers)

    # Use static arrays if there are not too many variables
    if num_vars < 100 && static_arrays
        if !(centers[1] isa StaticArray)
            centers = [ SVector{num_vars}(c) for c ∈ centers ]
        end
    end

    kernels = make_kernels( φ, centers )
    poly_deg = max( poly_deg, cpd_order(φ) - 1 , -1 )
    poly_basis = canonical_basis( num_vars, poly_deg )

    w, λ = coefficients( sites, values, kernels, poly_basis )

    # build output polynomials
    poly_vec = StaticPolynomials.Polynomial[]
    for coeff_ℓ ∈ eachcol( λ )
        push!( poly_vec, StaticPolynomials.Polynomial( poly_basis'coeff_ℓ ) )
    end
    poly_sys = PolynomialSystem( poly_vec... )

    # build RBF system
    num_centers = length(centers)
    rbf_sys = RBFSum(kernels, w, num_vars, num_centers, num_outputs; static_arrays)

    # vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : true

    return RBFModel{vec_output}( rbf_sys, poly_sys, num_vars, num_centers, num_outputs )
end

### Special Constructors
````

We offer some specialized models (that simply wrap the main type).

````@example RBFModels
struct RBFInterpolationModel
    model :: RBFModel
end
(mod :: RBFInterpolationModel)(args...) = mod.model(args...)
@forward RBFInterpolationModel.model grad, jac, jacT, auto_grad, auto_jac
````

The constructor is a tiny bit simpler and additional checks take place:

````@example RBFModels
"""
    RBFInterpolationModel(features, labels, φ, poly_deg; kwargs… )

Build a model interpolating the feature-label pairs.
Does not accept `center` keyword argument.
"""
function RBFInterpolationModel(
        features :: AnyVec{ <:NumberOrVector },
        labels :: AnyVec{ <:NumberOrVector },
        φ :: Union{RadialFunction,AnyVec{<:RadialFunction}} = Multiquadric(),
        poly_deg :: Int = 1;
        vector_output :: Bool = true,
        static_arrays :: Bool = true
    )
    @assert length(features) == length(labels) "Provide as many features as labels!"
    mod = RBFModel(features, labels, φ, poly_deg; vector_output, static_arrays)
    return RBFInterpolationModel( mod )
end
````

We want to provide a convenient alternative constructor for interpolation models
so that the radial function can be defined by passing a `Symbol` or `String`.

````@example RBFModels
const SymbolToRadialConstructor = NamedTuple((
    :gaussian => Gaussian,
    :multiquadric => Multiquadric,
    :inv_multiquadric => InverseMultiquadric,
    :cubic => Cubic,
    :thin_plate_spline => ThinPlateSpline
))

function RBFInterpolationModel(
        features :: AnyVec{ <:NumberOrVector },
        labels :: AnyVec{ <:NumberOrVector },
        φ_symb :: Union{Symbol, String},
        φ_args :: Union{Nothing, Tuple} = nothing,
        poly_deg :: Int = 1; kwargs...
    )
    # which radial function to use?
    radial_symb = Symbol( lowercase( string( φ_symb ) ) )
    if !(radial_symb ∈ keys(SymbolToRadialConstructor))
        @warn "Radial Funtion $(radial_symb) not known, using Gaussian."
        radial_symb = :gaussian
    end

    constructor = SymbolToRadialConstructor[radial_symb]
    if φ_args isa Tuple
        φ = constructor( φ_args... )
    else
        φ = constructor()
    end

    RBFInterpolationModel( features, labels, φ, poly_deg; kwargs... )
end
````

[^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
[^wendland]: “Scattered Data Approximation”, Wendland

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

