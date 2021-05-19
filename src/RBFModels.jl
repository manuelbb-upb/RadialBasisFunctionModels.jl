module RBFModels #src

export RBFModel, RBFInterpolationModel #src
export Multiquadric, InverseMultiquadric, Gaussian, Cubic, ThinPlateSpline #src

export auto_grad, auto_jac, grad, jac, eval_and_auto_grad
export eval_and_auto_jac, eval_and_grad, eval_and_jac

# Dependencies of this module: 
using StaticPolynomials 
using ThreadSafeDicts
using Memoization: @memoize
using StaticArrays
using LinearAlgebra: norm
using Lazy: @forward

import Flux.Zygote as Zyg
using Flux.Zygote: Buffer

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

# For the interpolation system to be solvable we have to choose the 
# right polynomial space for ``p``.
# Basically, if the RBF Kernel (or the radial function) is 
# *conditionally positive definite* of order ``D`` we have to 
# find a polynomial ``p`` with ``\deg p \ge D-1``.[^wendland]
# If the kernel is CPD of order ``D=0`` we do not have to add an polynomial 
# and can interpolate arbitrary (distinct) data points. \
# Now let ``\{p_j}_{1\le j\le Q}`` be a basis of the polynomial space.
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
# We can also use differing feature vectors and centers. It is also possible to 
# determine a least squarse solution to a overdetermined system.
# Hence, we will denote the number of kernel centers by ``N_c`` from now on.

# !!! note 
#     When we have vector data ``Y ⊂ ℝ^k``, e.g. from modelling MIMO functions, then 
#     Julia easily allows for multiple columns in the righthand side of the interpolation 
#     equation system and we get weight vectors for multiple models, that can 
#     be thought of as one vector models ``r\colon ℝ^n \to ℝ``.

# !!! note 
#     See the section about **Constructors** for how we actually solve the equation system.

# ## Radial Basis Function Sum.

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
# We also need the so called order of conditional positive definiteness:
cpd_order( φ :: RadialFunction) :: Int = nothing;
# The derivative can also be specified. It defaults to
df( φ :: RadialFunction, ρ ) = Zyg.gradient( φ, ρ )[1]

# The file `radial_funcs.jl` contains various radial function implementations.
include("radial_funcs.jl")

# From an `RadialFunction` and a vector we can define a shifted kernel function.
const NumberOrVector = Union{<:Real, AbstractVector{<:Real}}

struct ShiftedKernel{RT <: RadialFunction, CT <: AbstractVector{<:Real}} <: Function
    φ :: RT
    c :: CT
end

norm2( vec ) = norm(vec, 2)

"Evaluate kernel `k` at `x - k.c`."
function (k::ShiftedKernel)( x :: AbstractVector{<:Real} )
    return k.φ( norm2( x - k.c ) )
end

# A vector of ``N`` kernels is a mapping ``ℝ^n → ℝ^N, \ x ↦ [ k₁(x), …, k_N(x)] ``.
"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]`` at `x`."
function ( K::AbstractVector{<:ShiftedKernel})( x :: AbstractVector{<:Real} )
    [ k(x) for k ∈ K ]
end

# Suppose, we have calculated the distances ``\|x - x^i\|`` beforehand.
# We can save redundant effort by passing them to the radial fucntions of the kernels.

"Evaluate `k.φ` for distance `ρ` where `ρ` should equal `x - k.c` for the argument `x`."
eval_at_dist( k :: ShiftedKernel , ρ :: Real ) = k.φ(ρ)

"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]``, provided the distances ``[ ρ_1(x), …, ρ_{N_c}(x) ]``."
function eval_at_dist( K::AbstractVector{<:ShiftedKernel}, dists :: AbstractVector{<:Real})
    [ eval_at_dist(k,ρ) for (k,ρ) ∈ zip(K,dists) ]
end

# Provided we have solved the interpolation system, the weights for the radial basis function 
# part of ``r`` are ``w``, where ``w`` is a vector of length ``N_c`` or a matrix in ``ℝ^{N_c \times k}``
# where k is the number of outputs.
# We treat the general case ``k\ge 1`` and always assume ``w`` to be a matrix.

struct RBFSum{
    KT <: AbstractVector{<:ShiftedKernel},
    WT <: AbstractMatrix{<:Real}
}
    kernels :: KT
    weights :: WT # can be a normal matrix or a SMatrix
end

# Make it display nicely:
function Base.show( io :: IO, rbf :: RBFSum{KT,WT} ) where {KT, WT}
    compact = get(io, :compact, false)
    if compact 
        print(io, "RBFSum{$(KT), $(WT)}")
    else
        n_kernels, n_out = size(rbf.weights)
        print(io, "RBFSum\n")
        print(io, "* with $(n_kernels) kernels in an array of type $(KT)\n")
        print(io, "* and a $(n_kernels)×$(n_out) weight matrix of type $(WT).")
    end        
end

# We can easily evaluate the `ℓ`-th output of the `RBFPart`.
"Evaluate output `ℓ` of RBF sum `rbf::RBFSum`"
function (rbf :: RBFSum)(x :: AbstractVector{<:Real}, ℓ :: Int)
    (rbf.kernels(x)'rbf.weights[:,ℓ])[1]
end

# Use the above method for vector-valued evaluation of the whole sum:
"Evaluate `rbf::RBFSum` at `x`."
(rbf::RBFSum)( x :: AbstractVector{<:Real} ) = vec(rbf.kernels(x)'rbf.weights)

# As before, we allow to pass precalculated distance vectors:
function eval_at_dist( rbf::RBFSum, dists :: AbstractVector{<:Real}, ℓ :: Int ) 
   eval_at_dist( rbf.kernels, dists )'rbf.weights[:,ℓ]
end

function eval_at_dist( rbf :: RBFSum, dists :: AbstractVector{<:Real})
   vec(eval_at_dist(rbf.kernels, dists )'rbf.weights)
end

# For the PolynomialTail do something similar and 
# use a `StaticPolynomials.PolynomialSystem` with a weight matrix.
include("empty_poly_sys.jl")
struct PolySum{
        PS <: Union{EmptyPolySystem, PolynomialSystem},
        WT <: AbstractMatrix
    }
    polys :: PS
    weights :: WT       # n_out × n_polys matrix
    
    function PolySum( polys :: PS, weights :: WT) where{PS, WT}
        _, n_polys = size(weights)
        @assert npolynomials(polys) == n_polys "Number of polynomials does not macth."
        new{PS,WT}(polys, weights)
    end
end

(p :: PolySum)(x) = p.weights*p.polys(x)

# We now have all ingredients to define the model type.

"""
    RBFModel{V}

* `V` is `true` by default. It can be set to `false` only if the number 
  of outputs is 1. Then scalars are returned.

"""
struct RBFModel{V, 
        RS <: RBFSum, 
        PS <: PolySum }
    rbf :: RS
    polys :: PS

    ## Information fields
    num_vars :: Int
    num_outputs :: Int
    num_centers :: Int
end

# We want a model to be displayed in a sensible way:
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

# Evaluation is easy. We accept an additional `::Nothing` argument that does nothing 
# for now, but saves some typing later.
function vec_eval(mod :: RBFModel, x :: AbstractVector{<:Real}, :: Nothing)
    return mod.rbf(x) .+ mod.polys( x )
end

function scalar_eval(mod :: RBFModel, x :: AbstractVector{<:Real}, :: Nothing )
    return (mod.rbf(x) .+ mod.polys( x ))[1]
end

"Evaluate model `mod :: RBFModel` at vector `x`."
( mod :: RBFModel{true, RS, PS} where {RS,PS} )(x :: AbstractVector{<:Real}, ℓ :: Nothing = nothing ) = vec_eval(mod,x,ℓ)
( mod :: RBFModel{false, RS, PS} where {RS,PS} )(x :: AbstractVector{<:Real}, ℓ :: Nothing = nothing ) = scalar_eval(mod,x,ℓ)

"Evaluate scalar output `ℓ` of model `mod` at vector `x`."
function (mod :: RBFModel)( x :: AbstractVector{<:Real}, ℓ :: Int)
    return mod.rbf(x, ℓ) .+ mod.polys.polys[ℓ]( x )
end

## scalar input
const NothInt = Union{Nothing,Int}

function (mod :: RBFModel)(x :: Real, ℓ :: NothInt = nothing )
    @assert mod.num_vars == 1 "The model has more than 1 inputs. Provide a vector `x`, not a number."
    mod( [x,], ℓ) 
end
#include("derivatives.jl")
include("constructors.jl")

# [^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
# [^wendland]: “Scattered Data Approximation”, Wendland

end #src