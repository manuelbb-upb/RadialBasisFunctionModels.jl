module RBFModels #src

export RBFInterpolationModel #src
export Multiquadric, InverseMultiquadric, Gaussian, Cubic, ThinPlateSpline # src

using DynamicPolynomials  # src
using ThreadSafeDicts # src
using Memoize: @memoize # src 
using StaticArrays # src

using Flux.Zygote: Buffer
# TODO also set Flux.trainable to make inner parameters trainable

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
    # r( x^i ) \stackrel{!}= y^i \quad \text{for all $i=1,…,N$.}
# ```
# The function ``k(•) = φ(\|•\|_2)`` is radially symmetric around the origin.
# ``k`` is called the kernel of an RBF. 
#
# We define an abstract super type for radial functions:
abstract type RadialFunction <: Function end

# Each Type that inherits from `RadialFunction` should implement 
# an evaluation method:
(φ :: RadialFunction )( x :: Real ) :: Real = Nothing;

# From an `RadialFunction` and a vector we can define a shifted kernel function:
struct ShiftedKernel <: Function
    φ :: RadialFunction
    c :: Union{Vector, SVector}
end

norm2( vec ) = sqrt(sum( vec.^2 ))
function (k::ShiftedKernel)( x :: Union{Vector, SVector} )
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

struct Multiquadric <: RadialFunction
    α :: Real   # shape parameter 
    β :: Real   # exponent 

    Multiquadric(α = 1, β = .5) = begin 
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
struct InverseMultiquadric <: RadialFunction
    α :: Real 
    β :: Real 

    Multiquadric( α = 1, β = .5 ) = begin 
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
# find a polynomial ``p`` with ``\deg p = D-1``.[^wendland]
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
    basis = Monomial[]  # list of basis polynomials
    for d̄ = 0 : d 
        for multi_exponent ∈ non_negative_solutions( d̄, n )
            push!( basis, prod( Xvar .^ multi_exponent ) )
        end
    end
    return basis
end

# ### The equation system

# Set ``P = [ p_j(x^i) ] ∈ ℝ^{N × Q}`` and ``Φ = φ(\| x^i - x^j \|)``.
# The linear equation system for the coefficients of $r$ is 
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

function _rbf_matrix( kernels, sites )
    ## easy way:
    ## [ kernels[j](sites[i]) for i = eachindex(sites), j = eachindex(kernels) ]

    ## Zygote-compatible
    Φ = Buffer( sites[1], length(sites), length(kernels) )
    for (i, k) ∈ enumerate(kernels)
        Φ[:,i] = [ k( x ) for x ∈ sites ]
    end
    return copy(Φ)
end

function _poly_matrix( polys, sites)
    ## return [ polys[j](sites[i]) for i = eachindex(sites), j = eachindex(polys) ]
    P = Buffer( sites[1], length(sites), length(polys) )
    for (i, p) ∈ enumerate(polys)
        P[:, i] = [ p( x ) for x ∈ sites ]
    end
    return copy(P)
end

@doc """
    coefficients(sites, values, kernels, polys )

Return the coefficient matrices `w` and `λ` for an rbf model 
``r(x) = Σ_{i=1}^N wᵢ kᵢ(x) + Σ_{j=1}^M λᵢ pᵢ(x)``,
where ``N`` is the length of `kernels` and ``M``
is the length of `polys`.

The arguments are 
* an array of data sites `sites` with vector entries from ``ℝ^n``.
* an array of data values `values` with vector entries from ``ℝ^k``.
* an array `kernels` of RBF basis functions (of type `ShiftedKernel`)
* an array `polys` of polynomial basis functions.
"""
function coefficients( sites, values, kernels, polys )

    ## Φ-matrix, columns =̂ basis funcs, rows =̂ sites 
    N = length(sites);
    Φ = _rbf_matrix( kernels, sites )
    
    ## P-matrix, N × Q
    Q = length(polys)
    P = _poly_matrix( polys, sites )

    @show size(Φ)
    @show size(P)

    ## system matrix A
    Z = zeros( Q, Q )
    A = [ Φ  P;
          P' Z ];

    ## rhs
    F = transpose( hcat( values...) ) # vals to matrix, columns =̂ outputs
    RHS = [
        F ;
        zeros( eltype(eltype(values)), Q, size(F,2) )
    ];

    ## solve system
    coeff = A \ RHS 

    ## return w and λ
    return coeff[1 : N, :], coeff[N+1 : end, :]
end

# ### The Model Data Type

# We now have all ingredients to define the model type.
# We allow for vector valued data sites and determine multiple 
# outputs if needed.

# First, define some helper functions:

function convert_list_of_vecs( vec_type :: Type, list_of_vecs :: Vector{<:Union{Vector,SVector}} )
    return vec_type.(list_of_vecs)
end

## allow for providing scalar data
function convert_list_of_vecs( vec_type :: Type, list_of_vecs :: Vector{<:Real} )
    return convert_list_of_vecs( vec_type, [ [x,] for x ∈ list_of_vecs ] )
end

## do nothing if types alreay match
function convert_list_of_vecs(::Type{F}, list_of_vecs :: Vector{F} ) where F
    return list_of_vecs
end

"Return array of `ShiftedKernel`s based on `φ` with centers from `sites`."
function make_kernels( φ :: RadialFunction, sites :: Union{Vector, SVector} )
    return [ ShiftedKernel(φ, c) for c ∈ sites ]
end

"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `sites`."
function make_kernels( φ :: Vector{RadialFunction}, sites :: Union{Vector, SVector} )
    return [ ShiftedKernel(φ, c) for c ∈ sites ]
end

# The actual data type stores \
# ~~the (center) sites and data labels~~ \
# the coefficients and a (vector of) radial and polynomial basis function(s).
struct RBFInterpolationModel{S,F} # where{S<:Val, F<:AbstractFloat}
    w :: Union{SMatrix{<:Int, <:Int, F, <:Int}, Matrix{F}}    # RBF weight matrix
    λ :: Union{SMatrix{<:Int, <:Int, F, <:Int}, Matrix{F}}    # polynomial coefficient matrix
    kernels :: Vector{<:ShiftedKernel}   # vector of RBF basis functions 
    polys :: Vector{<:DynamicPolynomials.AbstractPolynomialLike} # polynomial basis functions

    num_vars :: Int
    num_outputs :: Int

    function RBFInterpolationModel(  
        Sites :: Vector{ VecTypeS },
        Values :: Vector{ VecTypeV },
        φ :: Union{RadialFunction,Vector{<:RadialFunction}},
        poly_deg :: Int,
        ) where { VecTypeS, VecTypeV }

        ## data integrity checks
        @assert length(Sites) == length(Values) "Provide as many data sites as data labels."
        @assert !isempty(Sites) "Provide at least 1 data site."
        num_vars = length(Sites[1])
        num_outputs = length(Values[1])
        @assert all( length(s) == num_vars for s ∈ Sites ) "All sites must have same dimension."
        @assert all( length(v) == num_outputs for v ∈ Values ) "All values must have same dimension."
        
        ## use same precision everywhere ( at least half-precision )
        TypeS = eltype( VecTypeS )
        TypeV = eltype( VecTypeV )
        dtype = promote_type( TypeS, TypeV, Float16 )

        ## use static arrays only if matrices are not big
        use_static = num_vars <= 10 && num_outputs <= 10

        NewVecTypeS = use_static ? SVector{ num_vars, dtype } : Vector{dtype}
        NewVecTypeV = use_static ? SVector{ num_outputs, dtype } : Vector{dtype}
        sites = convert_list_of_vecs( NewVecTypeS, Sites )
        values = convert_list_of_vecs( NewVecTypeV, Values )
        
        kernels = make_kernels( φ, sites )
        polys = canonical_basis( num_vars, poly_deg )

        w, λ = coefficients( sites, values, kernels, polys )
        
        new{use_static, dtype}( w, λ, kernels, polys, num_vars, num_outputs)
    end
end

"Evaluate `m` at vector `x`"
function (m :: RBFInterpolationModel{false,F} where F)( x :: Union{Vector, SVector} )
    vec( _rbf_matrix( m.kernels, [x,] ) * m.w + _poly_matrix( m.polys, [x,] ) ) * m.λ
end

"Evaluate `m` at vector `x`, using size information."
function (m :: RBFInterpolationModel{true,F} where F)( x :: Union{Vector, SVector} )
    vec( 
        SizedMatrix{ 1, length(m.kernels) }(_rbf_matrix( m.kernels, [x,] )) * m.w +
        SizedMatrix{ 1, length(m.polys) }(_poly_matrix( m.polys, [x,] ) ) * m.λ 
    )
end

# ## Derivatives 
# Assume that ``φ`` is two times continuously differentiable. \ 
# What is the gradient of an RBF model? 
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
# We have ``∂/∂x_i ξ(x) = x - x^j`` and thus
# ```math
    # ∇r(x) = \sum_{i=1}^N \frac{w_i φ\prime( \| x - x^i \| )}{\| x - x^i \|} (x - x^i) + ∇p(x)
# ```
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
#    \frac{ φ'( \left\| ξ \right\| )}{\|ξ\|} ξ_j = 
#    ξ_j 
#    \right)
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
# \dfrac{∂}{∂ξ_i} ξ_j = \begin{cases}
#     1 & \text{if $i = j$},\\
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
# ∇Ψ_j(0) = φ''(0) e^j.
# ```

# [^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
# [^wendland]: “Scattered Data Approximation”, Wendland

end #src