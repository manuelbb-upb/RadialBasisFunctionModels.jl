module RBFModels #src

export RBFModel, RBFInterpolationModel #src
export Multiquadric, InverseMultiquadric, Gaussian, Cubic, ThinPlateSpline #src

export auto_grad, auto_jac, grad, jac

# Dependencies of this module: 
import DynamicPolynomials as DP
using StaticPolynomials 
using ThreadSafeDicts
using Memoize: @memoize
using StaticArrays
using LinearAlgebra: norm
using Lazy: @forward

import Flux.Zygote as Zyg
#using Flux.Zygote: Buffer, @adjoint

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
# We allow evaluation for statically sized vectors, too:
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

# A vector of ``N`` kernels is a mapping ``ℝ^n → ℝ^N, \ x ↦ [ k₁(x), …, k_N(x)] ``.
"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]`` at `x`."
function ( K::AnyVec{ShiftedKernel})( x :: AnyVec{<:Real} )
    [ k(x) for k ∈ K ]
end

# Suppose, we have calculated the distances ``\|x - x^i\|`` beforehand.
# We can save redundant effort by passing them to the radial fucntions of the kernels.

"Evaluate `k.φ` for distance `ρ` where `ρ` should equal `x - k.c` for the argument `x`."
eval_at_dist( k :: ShiftedKernel , ρ :: Real ) = k.φ(ρ)

"Evaluate ``x ↦ [ k₁(x), …, k_{N_c}(x)]``, provided the distances ``[ ρ_1(x), …, ρ_{N_c}(x) ]``."
function eval_at_dist( K::AnyVec{ShiftedKernel}, dists :: Vector{<:Real})
    [ eval_at_dist(k,ρ) for (k,ρ) ∈ zip(K,dists) ]
end

# Provided we have solved the interpolation system, the weights for the radial basis function 
# part of ``r`` are ``w``, where ``w`` is a vector of length ``N_c`` or a matrix in ``ℝ^{N_c \times k}``
# where k is the number of outputs.
# We treat the general case ``k\ge 1`` and always assume ``w`` to be a matrix.

struct RBFSum
    kernels :: AnyVec{ShiftedKernel}
    weights :: AnyMat # weigth vectors, one for each output

    ## information fields 
    num_vars :: Int
    num_centers :: Int
    num_outputs :: Int

end

# We can easily evaluate the `ℓ`-th output of the `RBFPart`.
"Evaluate output `ℓ` of RBF sum `rbf::RBFSum`"
function (rbf :: RBFSum)(x :: AnyVec{<:Real}, ℓ :: Int)
    (rbf.kernels(x)'rbf.weights[:,ℓ])[1]
end

# Use the above method for vector-valued evaluation of the whole sum:
"Evaluate `rbf::RBFSum` at `x`."
(rbf::RBFSum)( x :: AnyVec{<:Real} ) = vec(rbf.kernels(x)'rbf.weights)

# As before, we allow to pass precalculated distance vectors:
function eval_at_dist( rbf::RBFSum, dists :: AnyVec{<:Real}, ℓ :: Int ) 
   eval_at_dist( rbf.kernels, dists )'rbf.weights
end

function eval_at_dist( rbf :: RBFSum, dists :: AnyVec{<:Real})
   vec(eval_at_dist(rbf.kernels, dists )'rbf.weights)
end

# For the PolynomialTail we use a `StaticPolynomials.PolynomialSystem`. \
# We now have all ingredients to define the model type.

"""
    RBFModel{V}

* `V` is `true` by default. It can be set to `false` only if the number 
  of outputs is 1. Then scalars are returned.

"""
struct RBFModel{V}
    rbf :: RBFSum
    polys :: PolynomialSystem

    ## Information fields
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

# We want a model to be displayed in a sensible way:
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

# Evaluation is easy. We accept an additional `::Nothing` argument that does nothing 
# for now, but saves some typing later.
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

## scalar input
const NothInt = Union{Nothing,Int}

function (mod :: RBFModel)(x :: Real, ℓ :: NothInt = nothing )
    @assert mod.num_vars == 1 "The model has more than 1 inputs. Provide a vector `x`, not a number."
    mod( [x,], ℓ) 
end

# ## Derivatives 

# The easiest way to provide derivatives is via Automatic Differentiation.
# We have imported `Flux.Zygote` as `Zyg`. 
# For automatic differentiation we need custom adjoints for some `StaticArrays`:
Zyg.@adjoint (T::Type{<:SizedMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SizedVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
# This allows us to define the following methods:

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

# !!! note
#     We need at least `ChainRules@v.0.7.64` to have `auto_grad` etc. work for StaticArrays,
#     see [this issue](https://github.com/FluxML/Zygote.jl/issues/860).

# !!! note 
#     The above methods do not work if x is a StaticArray due to StaticPolynomials not knowing the 
#     custom adjoints. Maybe extendings `ChainRulesCore` helps?
# 

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

# We can then implement the formula from above.
# For a fixed center ``x^i`` let ``o`` be the distance vector ``x - x^i`` 
# and let ``ρ`` be the norm ``ρ = \|o\| = \| x- x^i \|``.
# Then, the gradient of a single kernel is:
function grad( k :: ShiftedKernel, o :: AnyVec{<:Real}, ρ :: Real )
    ρ == 0 ? zero(k.c) : (df( k.φ, ρ )/ρ) .* o
end

# In terms of `x`:
function grad( k :: ShiftedKernel, x :: AnyVec{<:Real} ) 
    o = x - k.c     # offset vector 
    ρ = norm2( o )  # distance 
    return grad( k, o, ρ )
end 

# The jacobion of a vector of kernels follows suit:
function jacT( K :: AnyVec{ShiftedKernel}, x :: AnyVec{<:Real})
    hcat( ( grad(k,x) for k ∈ K )... )
end 
## precalculated offsets and distances, 1 per kernel
function jacT( K :: AnyVec{ShiftedKernel}, offsets :: AnyVec{<:AnyVec}, dists :: AnyVec{<:Real} )
    hcat( ( grad(k,o,ρ) for (k,o,ρ) ∈ zip(K,offsets,dists) )... )
end
jac( K :: AnyVec{ShiftedKernel}, args... ) = transpose( jacT(K, args...) )

# Hence, the gradients of an RBFSum are easy:
function grad( rbf :: RBFSum, x :: AnyVec{<:Real}, ℓ :: Int = 1 )
    vec( jacT( rbf.kernels, x) * rbf.weights[:,ℓ] )    
end

function grad( rbf :: RBFSum, offsets :: AnyVec{<:AnyVec}, dists :: AnyVec{<:Real}, ℓ :: Int)
    return vec( jacT( rbf.kernels, offsets, dists ) * rbf.weights[:,ℓ] )
end

function grad( mod :: RBFModel, x :: AnyVec{<:Real}, ℓ :: Int = 1 )
    grad(mod.rbf, x, ℓ) + gradient( mod.polys.polys[ℓ], x )
end

# We can exploit our custom evaluation methods for "distances": 
function eval_and_grad( rbf :: RBFSum, offsets :: AnyVec{<:AnyVec}, dists :: AnyVec{<:Real}, ℓ :: Int)
    return eval_at_dist( rbf, dists, ℓ ), grad( rbf, offsets, dists, ℓ)
end

function eval_and_grad( rbf :: RBFSum, x :: AnyVec{<:Real}, ℓ :: Int = 1)
    offsets = [ x - k.c for k ∈ rbf.kernels ]
    dists = norm2.(offsets)
    return eval_and_grad( rbf, offsets, dists, ℓ)
end

# For the jacobian, we use this trick to save evaluations, too.
function jacT( rbf :: RBFSum, x :: AnyVec{<:Real} )
    offsets = [ x - k.c for k ∈ rbf.kernels ]
    dists = norm2.(offsets)
    jacT( rbf.kernels, offsets, dists )*rbf.weights
end
jac(rbf :: RBFSum, args... ) = transpose( jacT(rbf, args...) )

function jac( mod :: RBFModel, x :: AnyVec{<:Real} )
    jac( mod.rbf, x) + jacobian( mod.polys, x )
end

# !!! note
#     Hessians are not yet implemented.

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

include("constructors.jl")

# [^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Wild, 2009.
# [^wendland]: “Scattered Data Approximation”, Wendland

end #src