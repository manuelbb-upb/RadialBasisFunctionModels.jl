
# ## Derivatives 

# The easiest way to provide derivatives is via Automatic Differentiation.
# We have imported `Zygote` as `Zyg`. 
# For automatic differentiation we need custom adjoints for some `StaticArrays`:
Zyg.@adjoint (T::Type{<:SizedMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SizedVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)
Zyg.@adjoint (T::Type{<:SArray})(x::AbstractArray) = T(x), dv -> (nothing, dv)
# This allows us to define the following methods:

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

# !!! note
#     We need at least `ChainRules@v.0.7.64` to have `auto_grad` etc. work for StaticArrays,
#     see [this issue](https://github.com/FluxML/Zygote.jl/issues/860).

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
function grad( k :: ShiftedKernel, o :: AbstractVector{<:Real}, ρ :: Real )
    ρ == 0 ? zero(k.c) : (df( k.φ, ρ )/ρ) .* o
end

# In terms of `x`:
function grad( k :: ShiftedKernel, x :: AbstractVector{<:Real} ) 
    o = x - k.c     # offset vector 
    ρ = norm2( o )  # distance 
    return grad( k, o, ρ )
end 

# The jacobion of a vector of kernels follows suit:
function jacT( K :: AbstractVector{<:ShiftedKernel}, x :: AbstractVector{<:Real})
    hcat( ( grad(k,x) for k ∈ K )... )
end 
## precalculated offsets and distances, 1 per kernel
function jacT( K :: AbstractVector{<:ShiftedKernel}, offsets :: AbstractVector{<:AbstractVector}, dists :: AbstractVector{<:Real} )
    hcat( ( grad(k,o,ρ) for (k,o,ρ) ∈ zip(K,offsets,dists) )... )
end
jac( K :: AbstractVector{<:ShiftedKernel}, args... ) = transpose( jacT(K, args...) )

# Hence, the gradients of an RBFSum are easy:
function grad( rbf :: RBFSum, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    #vec( jacT( rbf.kernels, x) * rbf.weights[:,ℓ] )    
    vec( rbf.weights[ℓ,:]'jac( rbf.kernels, x ) )
end

function grad( rbf :: RBFSum, offsets :: AbstractVector{<:AbstractVector}, dists :: AbstractVector{<:Real}, ℓ :: Int)
    return vec( rbf.weights[ℓ,:]'jac( rbf.kernels, offsets, dists ) )
end

# The `grad` method looks very similar for the `PolySum`.
# We obtain the jacobian of the polynomial basis system via 
# `PolynomialSystem.jacobian`.
function grad( psum :: PolySum, x :: AbstractVector{<:Real} , ℓ :: Int = 1)
    return vec( psum.weights[ℓ,:]'jacobian( psum.polys, x ))
end

# For the `RBFModel` we simply combine both methods:
function _grad( mod :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    return grad(mod.rbf, x, ℓ) + grad( mod.psum, x, ℓ )
end

function grad( mod :: RBFModel, x :: Vector{<:Real}, ℓ :: Int = 1 )
    G = _grad(mod, x, ℓ)
    
    if G isa Vector
        return G
    else
        return [ G.data... ]
    end
end

function grad( mod :: RBFModel, x :: StaticVector{T, R} where{T, R<:Real}, ℓ :: Int = 1 )
    G = _grad(mod, x, ℓ)
    
    if G isa StaticArray
        return G
    else
        return SizedVector{mod.num_vars}(G)
    end
end

# We can exploit our custom evaluation methods for "distances": 
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

# For the `PolySum` we use `evaluate_and_jacobian`.
function eval_and_grad( psum :: PolySum, x :: AbstractVector{<:Real}, ℓ :: Int = 1)
    res_p, J_p = evaluate_and_jacobian( psum.polys, x )
    return (psum.weights[ℓ,:]'res_p)[1], vec(psum.weights[ℓ,:]'J_p)
end

# Combine for `RBFModel`:
function eval_and_grad( mod :: RBFModel, x :: AbstractVector{<:Real}, ℓ :: Int = 1 )
    res_rbf, g_rbf = eval_and_grad( mod.rbf, x, ℓ )
    res_polys, g_polys = eval_and_grad( mod.psum, x, ℓ )
    return res_rbf + res_polys, g_rbf + g_polys
end

# For the jacobian, we use the same trick to save evaluations.
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

function jac( mod :: RBFModel, x :: Vector{R}) where R<:Real
    Matrix( _jac(mod, x) )
end

function jac( mod :: RBFModel, x :: StaticVector{T, R} ) where{T, R<:Real}
    J = _jac(mod, x)
    if J isa StaticArray 
        return J 
    else
        return SizedMatrix{mod.num_outputs, mod.num_vars}(J)
    end 
end

# As before, define an "evaluate-and-jacobian" function that saves evaluations:
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
