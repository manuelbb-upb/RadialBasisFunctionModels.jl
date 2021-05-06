```@meta
EditURL = "<unknown>/src/RBFModels.jl"
```

Dependencies of this module:

````@example RBFModels
using DynamicPolynomials, StaticPolynomials
using ThreadSafeDicts
using Memoize: @memoize
using StaticArrays

using Flux.Zygote: Buffer, @adjoint
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
The function ``k(•) = φ(\|•\|_2)`` is radially symmetric around the origin.
``k`` is called the kernel of an RBF.

We define an abstract super type for radial functions:

````@example RBFModels
abstract type RadialFunction <: Function end
````

Each Type that inherits from `RadialFunction` should implement
an evaluation method.
It takes the radius/distance ``ρ = ρ(x) = \| x - x^i \|`` from
``x` to a specific center ``x^i``.

````@example RBFModels
(φ :: RadialFunction )( ρ :: Real ) :: Real = Nothing;
nothing #hide
````

From an `RadialFunction` and a vector we can define a shifted kernel function.
We allow evaluation for statically sized vectors, too:

````@example RBFModels
const AnyVec{T} = Union{Vector{T}, SVector{I, T}, SizedVector{I,T,V}} where {I,V}

struct ShiftedKernel <: Function
    φ :: RadialFunction
    c :: AnyVec
end

norm2( vec ) = sqrt(sum( vec.^2 ))

"Evaluate kernel `k` at `x - k.c`."
function (k::ShiftedKernel)( x :: AnyVec )
    return k.φ( norm2( x .- k.c ) )
end
````

!!! note
    When we have vector data ``Y ⊂ ℝ^k``, e.g. from modelling MIMO functions, then
    Julia easily allows for multiple columns in the righthand side of the interpolation
    equation system and we get weight vectors for multiple models, that can
    be thought of as one vector models ``r\colon ℝ^n \to ℝ``.

## Some Radial Functions

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
````

The **Multiquadric** is ``φ(ρ) = - \sqrt{ 1 + (αρ)^2 }`` and also has a positive shape
parameter. We can actually generalize it to the following form:

````@example RBFModels
"""
    Multiquadric( α = 1, β = 1//2 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{ \\ceil{β} } ( 1 + (αρ)^2 )^β
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
````

The **Cubic** is ``φ(ρ) = ρ^3``.
It can also be generalized:

````@example RBFModels
"""
    Cubic( β = 3 ) <: RadialFunction

A `RadialFunction` with
```math
    φ(ρ) = (-1)^{ \\ceil{β}/2 } ρ^β
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
````

## Solving the Interpolation System

### Polynomial Tail

For the interpolation system to be solvable we have to choose the
right polynomial space for ``p``.
Basically, if the RBF Kernel (or the radial function) is
*conditionally positive definite* of order ``D`` we have to
find a polynomial ``p`` with ``\deg p \ge D-1``.[^wendland]
If the kernel is CPD of order ``D=0`` we do not have to add an polynomial
and can interpolate arbitrary (distinct) data points.

````@example RBFModels
cpd_order( :: Gaussian ) = 0
cpd_order( φ :: Multiquadric ) = ceil( Int, φ.β )
cpd_order( :: InverseMultiquadric ) = 0
cpd_order( φ :: Cubic ) = ceil( Int, φ.β/2 )
cpd_order( φ :: ThinPlateSpline ) = φ.k + 1
````

The dimension of ``Π_{d}(ℝ^n)``, the space of ``n``-variate polynomials of
degree at most ``d``, is
```math
   Q = \binom{n+d}{n}
```
which equates to ``Q = n+1`` for linear and ``Q = (n+2)(n+1)/2`` for
quadratic polynomials. \
We need ``\{p_j\}_{1 \le j \le Q}``, a basis of ``Π_d(ℝ^n)``.

The canonical basis is ``x_1^{α_1} x_2^{α_2} … x_n^{α_n}`` with
``α_i ≥ 0`` and ``Σ_i α_i ≤ d``.
For ``\bar{d} \le d`` we can recursively get the non-negative integer solutions for
``Σ_i α_i = \bar{d}`` with the following function:

````@example RBFModels
@doc """
    non_negative_solutions( d :: Int, n :: Int)

Return array of solution vectors [x_1, …, x_n] to the equation
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
    @polyvar Xvar[1 : n]
    basis = DynamicPolynomials.Polynomial{true,Int}[] # list of basis polynomials
    for d̄ = 0 : d
        for multi_exponent ∈ non_negative_solutions( d̄, n )
            push!( basis, DynamicPolynomials.Polynomial(prod( Xvar .^ multi_exponent ) ))
        end
    end
    return basis
end
````

### The equation system

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

We want a `coefficients` function and use the following helpers:

````@example RBFModels
"Evaluate each function in `funcs` on each number/vector in `func_args`,
so that each column corresponds to a function evaluation."
function _func_matrix( funcs, func_args )
    # easy way:
    ##[ funcs[j](func_args[i]) for i = eachindex(func_args), j = eachindex(funcs) ]

    # Zygote-compatible
    Φ = Buffer( func_args[1], length(func_args), length(funcs) )
    for (i, func) ∈ enumerate(funcs)
        Φ[:,i] = func.( func_args )
    end
    return copy(Φ)
end
````

For now, we use the `\` operator to solve `A * coeff = RHS`.
Furthermore, we allow for different interpolation `sites` and
RBF centers by allowing for passing `kernels`.

````@example RBFModels
@doc """
    coefficients(sites, values, centers, rad_funcs, polys )

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
    sites :: Vector{ST},
    values :: Vector{VT},
    kernels :: Vector{ShiftedKernel},
    polys :: Vector{<:DynamicPolynomials.Polynomial}
    ) where {ST,VT}

    n_out = length(values[1])

    # Φ-matrix, N columns =̂ basis funcs, rows =̂ sites
    N = length(kernels);
    Φ = _func_matrix( kernels, sites )

    # P-matrix, N × Q
    Q = length(polys)
    P = _func_matrix( polys, sites )

    # system matrix A
    Z = zeros( eltype(Φ), Q, Q )
    A = [ Φ  P;
          P' Z ];

    # build rhs (in a Zygote friendly way)
    F = Buffer( values[1], length(values), length(values[1]) ) ## vals to matrix, columns =̂ outputs
    for (i, val) ∈ enumerate(values)
        F[i, :] = val
    end
    RHS = [
        copy(F) ;
        zeros( eltype(eltype(values)), Q, size(F,2) )
    ];

    # solve system
    coeff = A \ RHS

    # return w and λ
    if ST <: SVector
        return SMatrix{N,n_out}(coeff[1 : N, :]), SMatrix{Q, n_out}(coeff[N+1 : end, :])
    else
        return coeff[1 : N, :], coeff[N+1 : end, :]
    end
end
````

### The Model Data Type

We now have all ingredients to define the model type.
We allow for vector valued data sites and determine multiple
outputs if needed.

First, define some helper functions:

````@example RBFModels
function convert_list_of_vecs( vec_type :: Type, list_of_vecs :: Vector{<:Union{Vector,SVector}} )
    return vec_type.(list_of_vecs)
end

# allow for providing scalar data
function convert_list_of_vecs( vec_type :: Type, list_of_vecs :: Vector{<:Real} )
    return convert_list_of_vecs( vec_type, [ [x,] for x ∈ list_of_vecs ] )
end

# do nothing if types alreay match
function convert_list_of_vecs(::Type{F}, list_of_vecs :: Vector{F} ) where F
    return list_of_vecs
end

"Return array of `ShiftedKernel`s based on `φ` with centers from `sites`."
function make_kernels( φ :: RadialFunction, sites :: AnyVec )
    return [ ShiftedKernel(φ, c) for c ∈ sites ]
end

"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `sites`."
function make_kernels( φ_arr :: Vector{RadialFunction}, sites :: AnyVec )
    @assert length(φ_arr) == length(sites) "Provide as many functions `φ_arr` as `sites`."
    return [ ShiftedKernel(φ[i], sites[i]) for i = eachindex( φ_arr ) ]
end
````

The actual data does *not* store the coefficients, but rather:
* a `RBFOutputSystem`s and
* a `PolynomialSystem` (~ vector of polynomials) with `num_outputs` entries.
This should proof beneficial for evaluation and differentiation.

````@example RBFModels
struct RBFOutputSystem{S}
    kernels :: Vector{ShiftedKernel}
    weights :: Union{Matrix, SMatrix, SizedMatrix}

    num_outputs :: Int
    num_centers :: Int
end

const NumberOrVector{T} = Union{Real,Vector{T}, SVector{I, T}} where I;

"Return a vector containing the distance of `x` to each kernel center of `RBFOutputSystem`."
function _distances( rbf ::  RBFOutputSystem, x :: NumberOrVector )
    return [ norm2( k.c .- x ) for k ∈ rbf.kernels ]
end

"Return the vector [ φ₁(ρ₁) … φₙ(ρₙ) ] = [ k1(x) … kn(x) ]"
function _kernel_vector( rbf :: RBFOutputSystem, ρ :: Vector{<:Real} )
    return [ rbf.kernels[i].φ(ρ[i]) for i = 1 : length(ρ) ]
end

# evaluate at distances, static StaticArrays are used
"Evaluate (output `ℓ` of) `rbf` by plugging in distance `ρ[i]` in radial function `o.kernels[i].φ`."
function _eval_rbfs_at_ρ( rbf ::  RBFOutputSystem{true}, ρ :: Vector{<:Real}, ℓ :: Union{Int,Nothing} = nothing )
    W, n_out = isnothing(ℓ) ? (rbf.weights, rbf.num_outputs) : (rbf.weights[:, ℓ],1)  # should be sized
    vec(SVector{rbf.num_centers}( _kernel_vector( rbf, ρ ) )'W)
end

# evaluate at distances, normal Vectors are used
function _eval_rbfs_at_ρ(rbf :: RBFOutputSystem{false}, ρ :: Vector{<:Real}, ℓ :: Union{Int,Nothing} = nothing )
    W = isnothing(ℓ) ? rbf.weights : rbf.weights[:, ℓ]
    vec(_kernel_vector(rbf, ρ)'W)
end

"Evaluate `rbf :: RBFOutputSystem` at site `x`."
function ( rbf ::  RBFOutputSystem )( x :: NumberOrVector, ℓ :: Union{Int,Nothing} )
    ρ = _distances( rbf, x )        # calculate distance vector
    return _eval_rbfs_at_ρ( rbf, ρ, ℓ ) # eval at distances
end

# called by RBFModel{S,true}, vector output
_eval_rbf_sys(  ::Val{true}, rbf :: RBFOutputSystem, x :: NumberOrVector, ℓ :: Union{Int,Nothing} = nothing ) = rbf(x,ℓ)
# called by RBFModel{S,false}, scalar output
_eval_rbf_sys( ::Val{false}, rbf :: RBFOutputSystem, x :: NumberOrVector, ℓ :: Union{Int,Nothing} = nothing ) = rbf(x,ℓ)[end]
````

For evaluating polynomials, we build our own `PolySystem`:
It contains a list of StaticPolynomials and a flag indicating a static return type.

````@example RBFModels
struct PolySystem{S}
    polys :: Vector{StaticPolynomials.Polynomial}
    num_outputs :: Int
end

_eval_polys( poly_sys :: PolySystem{false}, x :: AnyVec, ℓ :: Nothing ) = [ p(x) for p ∈ poly_sys.polys ]
_eval_polys( poly_sys :: PolySystem{true}, x :: AnyVec, ℓ :: Nothing ) = SVector{poly_sys.num_outputs}([ p(x) for p ∈ poly_sys.polys ])
_eval_polys( poly_sys :: PolySystem{false}, x :: AnyVec, ℓ :: Int ) = [ poly_sys.polys[ℓ](x) ]
_eval_polys( poly_sys :: PolySystem{true}, x :: AnyVec, ℓ :: Int ) = SVector{1}([ poly_sys.polys[ℓ](x) ])
# StaticPolynomials cannot handle scalar input for variable vectors:
_eval_polys( poly_sys :: PolySystem{false}, x :: Real, ℓ :: Union{Nothing, Int}) = _eval_polys(poly_sys,[x,],ℓ)
_eval_polys( poly_sys :: PolySystem{true}, x :: Real, ℓ :: Union{Nothing, Int}) = _eval_polys(poly_sys,SVector{1}([x,]),ℓ)

# called below, from RBFModel, vector output and scalar output
_eval_poly_sys( ::Val{true}, poly_sys, x, ℓ) = _eval_polys( poly_sys, x, ℓ)
_eval_poly_sys( ::Val{false}, poly_sys, x, ℓ) = _eval_polys( poly_sys, x, ℓ)[end]
````

The final model struct then is:

````@example RBFModels
"""
    RBFModel{S,V}

* `S` is `true` or `false` and indicates whether static arrays are used or not.
* `V` is `true` if vectors should be returned and `false` if scalars are returned.

Initialize via one of the constructors, e.g.,
    `RBFInterpolationModel( sites, values, φ, poly_deg )`
to obain an interpolating RBF model.

See also [`RBFInterpolationModel`](@ref)
"""
struct RBFModel{S,V}
    rbf_sys :: RBFOutputSystem{S}
    poly_sys :: PolySystem{S}

    # Information fields
    num_vars :: Int
    num_outputs :: Int
    num_centers :: Int
end

function (mod :: RBFModel{S,V} )( x :: NumberOrVector, ℓ :: Union{Nothing,Int} = nothing ) where{S,V}
    rbf_eval = _eval_rbf_sys( Val(V), mod.rbf_sys, x, ℓ )
    poly_eval = _eval_poly_sys( Val(V), mod.poly_sys, x, ℓ )

    return rbf_eval .+ poly_eval
end
````

The `RBFInterpolationModel` constructor takes data sites and values and return an `RBFModel` that
interpolates these points.
We allow for passing scalar data and transform it internally.

````@example RBFModels
"""
    RBFInterpolationModel( sites :: Vector{VS}, values :: Vector{VT}, φ, poly_deg = 1;
        static_arrays = nothing, vector_output = true ) where {VS<:NumberOrVector, VT<:NumberOrVector}

Return an RBFModel `m` that is interpolating, i.e., `m(sites[i]) == values[i]` for all
`i = eachindex(sites)`.
`φ` should be a `RadialFunction` or a vector of `RadialFunction`s that has the same length
as `sites` and `values`.
`poly_deg` specifies the degree of the multivariate polynomial added to the RBF model.
It will be reset if needed.
`static_arrays` is automatically set to `true` if unspecified and the data dimensions are small.
`vector_output` is ignored if the `values` have length > 1. Elsewise it specifies whether to return
vectors or scalars when evaluating.
"""
function RBFInterpolationModel(
    s̃ides :: Vector{ VecTypeS },
    ṽalues :: Vector{ VecTypeV },
    φ :: Union{RadialFunction,Vector{<:RadialFunction}},
    poly_deg :: Int = 1;
    static_arrays :: Union{Bool, Nothing} = nothing,
    vector_output :: Bool = true,
    ) where { VecTypeS<:NumberOrVector, VecTypeV<:NumberOrVector }

    # data integrity checks
    @assert length(s̃ides) == length(ṽalues) "Provide as many data sites as data labels."
    @assert !isempty(s̃ides) "Provide at least 1 data site."
    num_vars = length(s̃ides[1])
    num_outputs = length(ṽalues[1])
    @assert all( length(s) == num_vars for s ∈ s̃ides ) "All sites must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ ṽalues ) "All values must have same dimension."

    # use static arrays? if no user preference is set …
    if isnothing(static_arrays)
        # … use only if matrices are small
        static_arrays = (num_vars <= 10 && num_outputs <= 10)
    end

    # prepare provided training data
    # use same precision everywhere ( at least half-precision )
    TypeS = eltype( VecTypeS )
    TypeV = eltype( VecTypeV )
    dtype = promote_type( TypeS, TypeV, Float16 )
    NewVecTypeS = static_arrays ? SVector{ num_vars, dtype } : Vector{dtype}
    NewVecTypeV = static_arrays ? SVector{ num_outputs, dtype } : Vector{dtype}
    sites = convert_list_of_vecs( NewVecTypeS, s̃ides )
    values = convert_list_of_vecs( NewVecTypeV, ṽalues )

    kernels = make_kernels( φ, sites )
    poly_deg = min( poly_deg, cpd_order(φ) - 1 )
    poly_basis = canonical_basis( num_vars, poly_deg )

    w, λ = coefficients( sites, values, kernels, poly_basis )

    # build output polynomials
    poly_vec = StaticPolynomials.Polynomial[]
    for coeff_ℓ ∈ eachcol( λ )
        push!( poly_vec, StaticPolynomials.Polynomial( poly_basis'coeff_ℓ ) )
    end
    poly_sys = PolySystem{static_arrays}( poly_vec, num_outputs )

    # vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : false

    # build RBF system
    num_centers = length(sites)
    rbf_sys = RBFOutputSystem{static_arrays}(kernels, w, num_outputs, num_centers)

    return RBFModel{static_arrays, vec_output}(
        rbf_sys, poly_sys, num_vars, num_outputs, num_centers
    )
end
````

We want to provide an alternative constructor for interpolation models
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
        s̃ides :: Vector{ <: NumberOrVector },
        ṽalues :: Vector{ <:NumberOrVector },
        radial_func :: Union{Symbol, String},
        constructor_args :: Union{Nothing, Vector{<:Tuple}, Tuple} = nothing,
        poly_deg :: Int = 1; kwargs ...
    )

    # which radial function to use?
    radial_symb = Symbol( lowercase( string( radial_func ) ) )
    if !(radial_symb ∈ keys(SymbolToRadialConstructor))
        @warn "Radial Funtion $(radial_symb) not known, using Gaussian."
        radial_symb = :gaussian
    end
    constructor = SymbolToRadialConstructor[radial_symb]

    if isnothing(constructor_args)
        φ = constructor()
    elseif constructor_args isa Tuple
        φ = constructor( constructor_args... )
    elseif constructor_args isa Vector
        @assert length(constructor_args) == length(s̃ides)
        φ = [ constructor( arg_tuple... ) for arg_tuple ∈ constructor_args ]
    end

    return RBFInterpolationModel( s̃ides, ṽalues, φ, poly_deg; kwargs... )

end
````

## Derivatives
Assume that ``φ`` is two times continuously differentiable. \
What is the gradient of an RBF model?
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
We have ``\dfrac{∂}{∂x_i} ξ(x) = x - x^j`` and thus
```math
∇r(x) = \sum_{i=1}^N \frac{w_i φ\prime( \| x - x^i \| )}{\| x - x^i \|} (x - x^i) + ∇p(x)
```
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

### Custom Adjoints
For automatic differentiation we need custom adjoints for some `StaticArrays`:

````@example RBFModels
@adjoint (T::Type{<:StaticArrays.SizedMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
@adjoint (T::Type{<:StaticArrays.SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
````

[^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
[^wendland]: “Scattered Data Approximation”, Wendland

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

