module RBFModels #src

export RBFModel, RBFInterpolationModel #src
export Multiquadric, InverseMultiquadric, Gaussian, Cubic, ThinPlateSpline #src

export auto_grad, auto_jac, grad, jac

# Dependencies of this module: 
using DynamicPolynomials, StaticPolynomials 
using ThreadSafeDicts
using Memoize: @memoize
using StaticArrays

import Flux.Zygote as Zyg
using Flux.Zygote: Buffer, @adjoint

# TODO also set Flux.trainable to make inner parameters trainable #src

# # Radial Basis Function Models 

# The sub-module `RBFModels` provides utilities to work with radial 
# basis function [RBF] models.  
# Given ``N`` data sites ``X = \{ x^1, …, x^N \} ⊂ ℝ^n`` and values 
# ``Y = \{ y^1, …, y^N \} ⊂ ℝ``, an interpolating RBF model ``r\colon ℝ^n → ℝ`` 
# has the form 
# ```math 
    # r(x) = \sum_{i=1}^N w_i φ( \| x - x^i \|_2 ) + p(x),
# ```
# where `p` is a multivariate polynomial. 
# The radial function ``φ\colon [0, ∞) \to ℝ`` defines the RBF and we can solve for 
# the coefficients ``w`` by solving the interpolation system 
# ```math 
    # r( x^i ) \stackrel{!}= y^i \quad \text{for all }i=1,…,N
# ```
# The function ``k(•) = φ(\|•\|_2)`` is radially symmetric around the origin.
# ``k`` is called the kernel of an RBF. 
#
# We define an abstract super type for radial functions:
abstract type RadialFunction <: Function end

# Each Type that inherits from `RadialFunction` should implement 
# an evaluation method.
# It takes the radius/distance ``ρ = ρ(x) = \| x - x^i \|`` from 
# ``x`` to a specific center ``x^i``.
(φ :: RadialFunction )( ρ :: Real ) :: Real = Nothing;

# From an `RadialFunction` and a vector we can define a shifted kernel function.
# We allow evaluation for statically sized vectors, too:
const StatVec{T} = Union{SVector{I,T}, SizedVector{I,T,V}} where {I,V}
const AnyVec{T} = Union{Vector{T}, StatVec{T}}

struct ShiftedKernel <: Function
    φ :: RadialFunction
    c :: AnyVec 
end

norm2( vec ) = sqrt(sum( vec.^2 ))

"Evaluate kernel `k` at `x - k.c`."
function (k::ShiftedKernel)( x :: AnyVec )
    return k.φ( norm2( x .- k.c ) )
end

# !!! note 
#     When we have vector data ``Y ⊂ ℝ^k``, e.g. from modelling MIMO functions, then 
#     Julia easily allows for multiple columns in the righthand side of the interpolation 
#     equation system and we get weight vectors for multiple models, that can 
#     be thought of as one vector models ``r\colon ℝ^n \to ℝ``.

# ## Some Radial Functions 

# The **Gaussian** is defined by ``φ(ρ) = \exp \left( - (αρ)^2 \right)``, where 
# ``α`` is a shape parameter to fine-tune the function.

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

# The **Multiquadric** is ``φ(ρ) = - \sqrt{ 1 + (αρ)^2 }`` and also has a positive shape 
# parameter. We can actually generalize it to the following form:

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

# Related is the **Inverse Multiquadric** `` φ(ρ) = (1+(αρ)^2)^{-β}`` is related:
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

# The **Cubic** is ``φ(ρ) = ρ^3``. 
# It can also be generalized: 
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

# The thin plate spline is usually defined via 
# ``φ(ρ) = ρ^2 \log( ρ )``. 
# We provide a generalized version, which defaults to 
# ``φ(ρ) = - ρ^4 \log( ρ )``.
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

# ## Solving the Interpolation System 

# ### Polynomial Tail
# 
# For the interpolation system to be solvable we have to choose the 
# right polynomial space for ``p``.
# Basically, if the RBF Kernel (or the radial function) is 
# *conditionally positive definite* of order ``D`` we have to 
# find a polynomial ``p`` with ``\deg p \ge D-1``.[^wendland]
# If the kernel is CPD of order ``D=0`` we do not have to add an polynomial 
# and can interpolate arbitrary (distinct) data points.

cpd_order( :: Gaussian ) = 0 
cpd_order( φ :: Multiquadric ) = ceil( Int, φ.β ) 
cpd_order( :: InverseMultiquadric ) = 0
cpd_order( φ :: Cubic ) = ceil( Int, φ.β/2 )
cpd_order( φ :: ThinPlateSpline ) = φ.k + 1

# The dimension of ``Π_{d}(ℝ^n)``, the space of ``n``-variate polynomials of 
# degree at most ``d``, is 
# ```math 
#    Q = \binom{n+d}{n}
# ```
# which equates to ``Q = n+1`` for linear and ``Q = (n+2)(n+1)/2`` for 
# quadratic polynomials. \
# We need ``\{p_j\}_{1 \le j \le Q}``, a basis of ``Π_d(ℝ^n)``.

# The canonical basis is ``x_1^{α_1} x_2^{α_2} … x_n^{α_n}`` with 
# ``α_i ≥ 0`` and ``Σ_i α_i ≤ d``.
# For ``\bar{d} \le d`` we can recursively get the non-negative integer solutions for 
# ``Σ_i α_i = \bar{d}`` with the following function:

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
            ## make RHS smaller by and find all solutions of length `n-1`
            ## then concatenate with difference `d-i`
            for shorter_solution ∈ non_negative_solutions( i, n - 1)
                push!( solutions, [ d-i ; shorter_solution ] )
            end
        end
        return solutions
    end
end

# We use `DynamicPolynomials.jl` to generate the Polyomials.
# Furthermore, we employ Memoization (via `Memoize.jl` and `ThreadSafeDicts`)
# to save the result for successive usage.
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

# ### The equation system

# Set ``P = [ p_j(x^i) ] ∈ ℝ^{N × Q}`` and ``Φ = φ(\| x^i - x^j \|)``.
# In case of interpolation, the linear equation system for the coefficients of $r$ is 
# ```math 
#     \begin{bmatrix}
#     Φ & P \\
#     P^T & 0_{Q × Q}
#     \end{bmatrix}
#     \begin{bmatrix}
#         w \\
#         λ
#     \end{bmatrix}
#     = 
#     \begin{bmatrix}
#     Y 
#     \\ 
#     0_Q
#     \end{bmatrix}.
# ```

# We want a `coefficients` function and use the following helpers:

"Evaluate each function in `funcs` on each number/vector in `func_args`, 
so that each column corresponds to a function evaluation."
function _func_matrix( funcs, func_args )
    ## easy way:
    ##[ funcs[j](func_args[i]) for i = eachindex(func_args), j = eachindex(funcs) ]
    
    ## Zygote-compatible
    Φ = Buffer( func_args[1], length(func_args), length(funcs) )
    for (i, func) ∈ enumerate(funcs)
        Φ[:,i] = func.( func_args )
    end
    return copy(Φ)
end

# For now, we use the `\` operator to solve `A * coeff = RHS`.
# Furthermore, we allow for different interpolation `sites` and 
# RBF centers by allowing for passing `kernels`.

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
    kernels :: AnyVec{ShiftedKernel},
    polys :: Vector{<:DynamicPolynomials.Polynomial} 
    ) where {ST,VT}

    n_out = length(values[1])
    
    ## Φ-matrix, N columns =̂ basis funcs, rows =̂ sites
    N = length(kernels);
    Φ = _func_matrix( kernels, sites )
    
    ## P-matrix, N × Q
    Q = length(polys)
    P = _func_matrix( polys, sites )

    ## system matrix A
    Z = zeros( eltype(Φ), Q, Q )
    A = [ Φ  P;
          P' Z ];

    ## build rhs (in a Zygote friendly way)
    F = Buffer( values[1], length(values), length(values[1]) ) ## vals to matrix, columns =̂ outputs
    for (i, val) ∈ enumerate(values)
        F[i, :] = val
    end
    RHS = [
        copy(F) ;
        zeros( eltype(eltype(values)), Q, size(F,2) )
    ];

    ## solve system
    coeff = A \ RHS 

    ## return w and λ
    if ST <: SVector
        return SMatrix{N,n_out}(coeff[1 : N, :]), SMatrix{Q, n_out}(coeff[N+1 : end, :])
    else
        return coeff[1 : N, :], coeff[N+1 : end, :]
    end
end

# ### The Model Data Type

# The actual model type does *not* store the coefficients, but rather:
# * a `RBFOutputSystem`s and
# * a `PolynomialSystem` (~ vector of polynomials) with `num_outputs` entries.

struct RBFOutputSystem{S}
    kernels :: AnyVec{ShiftedKernel}
    weights :: Union{Matrix, SMatrix, SizedMatrix}

    num_outputs :: Int 
    num_centers :: Int
end

const NumberOrVector = Union{<:Real, AnyVec{<:Real}}


# We provide methods to evaluate the RBF part of a model, that can also 
# take the distance vector ``ρ(x) = [\| x - x^1 \|, …, \| x - x^N \|]``.
# This way, we can save evaluations later, when we evaluate and differentiate 
# with the same function. \
# There are a few helper functions:
"Return the list of difference vectors ``[x .- x^1, …, x - x^N]`` where ``x^i`` are the kernel centers of `rbf`."
function _offsets( rbf :: RBFOutputSystem, x :: AnyVec )
    return [ k.c .- x for k ∈ rbf.kernels ]
end

"Return a vector containing the distance of `x` to each kernel center of `RBFOutputSystem`."
function _distances( rbf ::  RBFOutputSystem, x :: AnyVec ) 
    return norm2.(_offsets( rbf, x) )
end

"Return the vector [ φ₁(ρ₁) … φₙ(ρₙ) ] = [ k1(x) … kn(x) ]"
function _kernel_vector( rbf :: RBFOutputSystem, ρ :: AnyVec{<:Real} )
    map.( getfield.(rbf.kernels, :φ), ρ)
end

## Define vectorization for a scalar.
Base.vec( x :: Number ) = [x,]

# The final method looks like this:
"Evaluate (output `ℓ` of) `rbf` by plugging in distance `ρ[i]` in radial function `o.kernels[i].φ`."
function _eval_rbfs_at_ρ(rbf :: RBFOutputSystem, ρ :: AnyVec{<:Real}, ℓ :: Union{Int,Nothing} = nothing ) 
    W = isnothing(ℓ) ? rbf.weights : rbf.weights[:, ℓ]
    vec(_kernel_vector(rbf, ρ)'W)
end

# Of course, it is easy to evaluate, when only the argument `x` is provided.
"Evaluate `rbf :: RBFOutputSystem` at site `x`. A single output can be specified with `ℓ`."
function ( rbf ::  RBFOutputSystem )( x :: AnyVec, ℓ :: Union{Int,Nothing} )
    ρ = _distances( rbf, x )        # calculate distance vector 
    return _eval_rbfs_at_ρ( rbf, ρ, ℓ ) # eval at distances 
end

# These methods are used internally:
## called by RBFModel{S,true}, vector output 
_eval_rbf_sys(  ::Val{true}, rbf :: RBFOutputSystem, x :: AnyVec, ℓ :: Union{Int,Nothing} = nothing ) = rbf(x,ℓ)
## called by RBFModel{S,false}, scalar output
_eval_rbf_sys( ::Val{false}, rbf :: RBFOutputSystem, x :: AnyVec, ℓ :: Union{Int,Nothing} = nothing ) = rbf(x,ℓ)[end]

# For evaluating polynomials, we build our own `PolySystem`: 
# It contains a list of StaticPolynomials and a flag indicating a static return type.

struct PolySystem{S}
    polys :: AnyVec{<:StaticPolynomials.Polynomial}
    num_outputs :: Int

    function PolySystem{S}( polys :: AnyVec, num_outputs ) where S
        @assert length(polys) == num_outputs "Provide as many polynomials as outputs."
        if S == true && !( polys isa StatVec )
            polys = SizedVector{num_outputs}(polys)
        end 
        return new{S}(polys,num_outputs)
    end
end

function ( poly_sys :: PolySystem)( x :: AnyVec, ℓ :: Nothing )  
    [ p(x) for p ∈ poly_sys.polys ]
end
function ( poly_sys :: PolySystem )( x :: AnyVec, ℓ :: Int ) 
    [ poly_sys.polys[ℓ](x),]
end

## called below, from RBFModel, vector output and scalar output
const NothInt = Union{Nothing, Int}
_eval_poly_sys( ::Val{true}, poly_sys :: PolySystem, x :: AnyVec , ℓ :: NothInt ) = poly_sys( x, ℓ)
_eval_poly_sys( ::Val{false}, poly_sys :: PolySystem, x :: AnyVec, ℓ :: NothInt ) = poly_sys( x, ℓ)[end]

# We now have all ingredients to define the model type.
# We allow for vector valued data sites and determine multiple 
# outputs if needed.

# First, define some helper functions to redefine the training data internally:

"Return a list of the elements of type `vec_type` applied to all elements from `list_of_vecs`."
function convert_list_of_vecs( vec_type :: Type, list_of_vecs :: Vector{<:AnyVec} )
    return vec_type.(list_of_vecs)
end

## allow for providing scalar data
function convert_list_of_vecs( vec_type :: Type, list_of_nums :: Vector{<:Real} )
    return convert_list_of_vecs( vec_type, [ [x,] for x ∈ list_of_nums ] )
end

## do nothing if types alreay match
function convert_list_of_vecs(::Type{F}, list_of_vecs :: Vector{F} ) where F
    return list_of_vecs
end

# Helpers to create kernel functions. Should return `SVector` when appropriate. 

function _make_kernels( φ_arr :: Union{RadialFunction, AnyVec{RadialFunction}}, sites :: Vector )
    if φ_arr isa RadialFunction
        φ_arr = [φ_arr for i = 1:length(sites)]
    end
    return [ ShiftedKernel(φ_arr[i], sites[i]) for i = eachindex( φ_arr ) ]
end

"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `sites`."
make_kernels( φ_arr, sites :: Vector{<:Vector} ) = _make_kernels( φ_arr, sites )
make_kernels( φ_arr, sites :: Vector{<:StatVec} ) = SVector{length(sites)}(_make_kernels(φ_arr,sites))

# The final model struct then is:

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

    ## Information fields
    num_vars :: Int
    num_outputs :: Int
    num_centers :: Int
end

function Base.show( io :: IO, mod :: RBFModel{S,V} ) where {S,V}
    compact = get(io, :compact, false)
    if compact 
        print(io, "$(mod.num_vars)D$(mod.num_outputs)D-RBFModel{$(S),$(V)")
    else
        print(io, "RBFModel\n")
        if S print(io, "* using StaticArrays\n") end 
        if V 
            print(io, "* with vector output\n") 
        else 
            print(io, "* with scalar output\n") 
        end
        print(io, "* with $(mod.num_centers) centers\n")
        print(io, "* mapping from ℝ^$(mod.num_vars) to ℝ^$(mod.num_outputs).")
    end        
end


function (mod :: RBFModel{S,V} )( x :: AnyVec, ℓ :: NothInt = nothing ) where{S,V} 
    rbf_eval = _eval_rbf_sys( Val(V), mod.rbf_sys, x, ℓ )
    poly_eval = _eval_poly_sys( Val(V), mod.poly_sys, x, ℓ ) 

    return rbf_eval .+ poly_eval
end

## scalar input
function (mod :: RBFModel)(x :: Real, ℓ :: NothInt = nothing )
    @assert mod.num_vars == 1 "The model has more than 1 inputs. Provide a vector `x`, not a number."
    mod( [x,], ℓ) 
end

# The `RBFInterpolationModel` constructor takes data sites and values and return an `RBFModel` that 
# interpolates these points.
# We allow for passing scalar data and transform it internally.

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

    ## data integrity checks
    @assert length(s̃ides) == length(ṽalues) "Provide as many data sites as data labels."
    @assert !isempty(s̃ides) "Provide at least 1 data site."
    num_vars = length(s̃ides[1])
    num_outputs = length(ṽalues[1])
    @assert all( length(s) == num_vars for s ∈ s̃ides ) "All sites must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ ṽalues ) "All values must have same dimension."
    
    ## use static arrays? if no user preference is set …
    if isnothing(static_arrays)
        ## … use only if matrices are small
        static_arrays = (num_vars <= 10 && num_outputs <= 10)
    end

    ## prepare provided training data
    ## use same precision everywhere ( at least half-precision )
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

    ## build output polynomials
    poly_vec = StaticPolynomials.Polynomial[] 
    for coeff_ℓ ∈ eachcol( λ )
        push!( poly_vec, StaticPolynomials.Polynomial( poly_basis'coeff_ℓ ) )
    end 
    poly_sys = PolySystem{static_arrays}( poly_vec, num_outputs )

    ## vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : true
    
    ## build RBF system 
    num_centers = length(sites)
    rbf_sys = RBFOutputSystem{static_arrays}(kernels, w, num_outputs, num_centers)
   
    return RBFModel{static_arrays, vec_output}( 
        rbf_sys, poly_sys, num_vars, num_outputs, num_centers
    )
end

# We want to provide an alternative constructor for interpolation models 
# so that the radial function can be defined by passing a `Symbol` or `String`.

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

    ## which radial function to use?
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

# ## Derivatives 

# The easiest way to provide derivatives is via Automatic Differentiation.
# We have imported `Flux.Zygote` as `Zyg.` this allows us to define the following methods:

# * A function to return the jacobian:
"Return the jacobian of `rbf` at `x` (using Zygote)."
function auto_jac( rbf :: RBFModel, x :: NumberOrVector )
    Zyg.jacobian( rbf, x )[1]
end

# * A function to evaluate the model and return the jacobian at the same time:
function eval_and_auto_jac( rbf :: RBFModel, x :: NumberOrVector )
    y, back = Zyg._pullback( rbf, x )

    T = eltype(y)
    n = length(y)
    jac = zeros(T, n, length(x) )
    for i = 1 : length(x)
        e = [ zeros(T, i -1 ); T(1); zeros(T, n - i )  ]
        jac[i, :] .= back(e)[2]
    end

    return y, jac
end

# * A function to return the gradient of a specific output: 
"Return gradient of output `ℓ` of model `rbf` at point `x` (using Zygote)."
function auto_grad( rbf :: RBFModel, x :: NumberOrVector, ℓ :: Union{Int,Nothing} = nothing)
    Zyg.gradient( χ -> rbf(χ, ℓ)[end], x )[1]
end

# * A function to evaluate the function and return the gradient 
function eval_and_auto_grad( rbf :: RBFModel, x :: NumberOrVector, ℓ :: Union{Int,Nothing} = nothing )
    y, back = Zyg._pullback( χ -> rbf(χ, ℓ)[end], x)

    grad = back( one(y) )[2]
    return y, grad
end

# !!! note 
#     The above methods will fail, if `x` is one of the rbf centers.


# But we don't need `Zygote`, because we can derive the gradients ourselves.
# Assume that ``φ`` is two times continuously differentiable. \ 
# What is the gradient of a scalar RBF model? 
# Using the chain rule and ``ξ = x - x^j`` we get 
# ```math 
    # \dfrac{∂}{∂ξ_i} \left( φ(\| ξ \|) \right)
    # = 
    # φ\prime ( \| ξ \| ) \cdot 
    # \dfrac{∂}{∂ξ_i} ( \| ξ \| )
    # = 
    # φ\prime ( \| ξ \| ) \cdot
    # \dfrac{ξ_i}{\|ξ\|}.
# ```
# The right term is always bounded, but not well defined for ``ξ = 0`` 
# (see [^wild_diss] for details). \
# **That is why we require ``φ'(0) \stackrel{!}= 0``.** \
# We have ``\dfrac{∂}{∂x_i} ξ(x) = 1`` and thus
# ```math
    # ∇r(x) = \sum_{i=1}^N \frac{w_i φ\prime( \| x - x^i \| )}{\| x - x^i \|} (x - x^i) + ∇p(x)
# ```

# Hence, we need the derivatives of our `RadialFunctions`.

df(φ :: Gaussian, ρ :: Real) = - 2 * φ.α * φ( ρ )
df(φ :: Multiquadric, ρ :: Real ) = (-1)^(ceil(Int, φ.β)) * 2 * φ.α * φ.β * ρ * ( 1 + (φ.α * ρ)^2 )^(φ.β - 1)
df(φ :: InverseMultiquadric, ρ :: Real ) = - 2 * φ.α^2 * φ.β * ρ * ( 1 + (φ.α * ρ)^2 )^(-φ.β - 1)
df(φ :: Cubic, ρ :: Real ) = (-1)^(ceil(Int, φ.β/2)) * φ.β * ρ^(φ.β - 1)
df(φ :: ThinPlateSpline, ρ :: Real ) = ρ == 0 ? 0 : (-1)^(φ.k+1) * ρ^(2*φ.k - 1) * ( 2 * φ.k * log(ρ) + 1)

# We can then implement the formula from above.
# For a fixed center ``x^i`` let ``o`` be the distance vector ``x - x^i`` 
# and let ``ρ`` be the norm ``ρ = \|o\| = \| x- x^i \|``.
# Then, the gradient of a single kernel is:
function grad( k :: ShiftedKernel, o :: AnyVec, ρ :: Real )
    ρ == 0 ? zero(k.c) : (df( k.φ, ρ )/ρ) .* o
end

# In terms of `x`:
function grad( k :: ShiftedKernel, x :: NumberOrVector ) 
    o = x .- k.c    # offset vector 
    ρ = norm2( o )  # distance 
    return grad( k, o, ρ )
end 

# Hence the gradients of an RBFOutputSystem are easy:
function grad( rbf :: RBFOutputSystem, x :: NumberOrVector, ℓ :: Int )
    W = rbf.weights[:, ℓ]   # weights for output ℓ 
    return sum( W .* [ grad(k, x ) for k ∈ kernels ] )
end

# For the jacobian, we use `grad(k, o, ρ)` to save evaluations:
function jac( rbf :: RBFOutputSystem{S}, x :: NumberOrVector ) where S
    
    T = promote_type( eltype(x), eltype(rbf.weights) )
    J = (S ? MMatrix{rbf.num_outputs, length(x)} : Matrix )( zeros(T, rbf.num_outputs, length(x) ) ) 

    o = [ x .- k.c for k ∈ rbf.kernels ]    # all offset vectors 
    ρ = norm2.(o)       # all distances 
    for ℓ = 1 : rbf.num_outputs
        W = rbf.weights[:,ℓ]
        J[ℓ, :] = sum( W .* [ grad( i...) for i ∈ zip( rbf.kernels, o, ρ) ] )
    end
    return J
end


# For the Hessian ``Hr \colon ℝ^n \to ℝ^{n\times n}`` we need the gradients of the 
# component functions 
# ```math 
#     ψ_j(ξ) = \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|} ξ_j 
# ```
# Suppose ``ξ ≠ 0``.
# First, using the product rule, we have 
# ```math 
#    \dfrac{∂}{∂ξ_i} 
#    \left( 
#    \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|} ξ_j  
#    \right) =
#    ξ_j 
#    \dfrac{∂}{∂ξ_i} 
#    \left( 
#    \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|}       
#    \right) 
#    + 
#    \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|}       
#    \dfrac{∂}{∂ξ_i} 
#    ξ_j 
# ```
# The last term is easy because of 
# ```math 
# \frac{∂}{∂ξ_i} ξ_j 
# = 
# \begin{cases}
#     1 & \text{if }i = j,\\
#     0 & \text{else.}
# \end{cases}
# ```
# For the first term we find 
# ```math 
#    \dfrac{∂}{∂ξ_i}   
#    \left( 
#      \frac{ φ'( \left\| ξ \right\| )}
#       {\|ξ\|}       
#    \right)
#    =
#    \frac{ 
#        φ'\left(\left\| ξ \right\|\right) ∂_i \|ξ\| 
#        - \|ξ\| ∂_i φ'\left( \left\| ξ \right\|\right) 
#     }{
#         \|ξ\|^2
#     }
#     = 
#     \frac{ 
#         \dfrac{φ'(\|ξ\|)}{\|ξ\|} ξ_i - \|ξ\|φ''(\|ξ\|)\dfrac{ξ_i}{\|ξ\|}  
#     }{\|ξ\|^2}
# ```
# Hence, the gradient of ``ψ_j`` is 
# ```math 
#     ∇ψ_j(ξ) 
#     = 
#     \left( \frac{φ'(\|ξ\|)}{\|ξ\|^3} 
#     -
#     \frac{φ''(\|ξ\|)}{\|ξ\|^2} \right) \cdot ξ 
#     -\frac{φ'(\|ξ\|)}{\|ξ\|} e^j,
# ```
# where ``e^j ∈ ℝ^n`` is all zeros, except ``e^j_j = 1``.
# For ``ξ = 0`` the first term vanishes due to L'Hôpital's rule:
# ```math 
# ∇ψ_j(0) = φ''(0) e^j.
# ```

# ### Custom Adjoints
# For automatic differentiation we need custom adjoints for some `StaticArrays`:
#@adjoint (T::Type{<:StaticArrays.SizedMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
#@adjoint (T::Type{<:StaticArrays.SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
#@adjoint (T::Type{<:StaticArrays.SizedVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)

# [^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
# [^wendland]: “Scattered Data Approximation”, Wendland

end #src