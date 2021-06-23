# # Some Radial Functions

using Parameters: @with_kw

#%%
# The **Gaussian** is defined by ``φ(ρ) = \exp \left( - (αρ)^2 \right)``, where 
# ``α`` is a shape parameter to fine-tune the function.

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

# The **Multiquadric** is ``φ(ρ) = - \sqrt{ 1 + (αρ)^2 }`` and also has a positive shape 
# parameter. We can actually generalize it to the following form:

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

# Related is the **Inverse Multiquadric** `` φ(ρ) = (1+(αρ)^2)^{-β}`` is related:
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

# The **Cubic** is ``φ(ρ) = ρ^3``. 
# It can also be generalized: 
"""
    Cubic( β = 3 ) <: RadialFunction

A `RadialFunction` with 
```math 
    φ(ρ) = (-1)^{ \\lceil β \\rceil /2 } ρ^β
```
"""
@with_kw struct Cubic{R<:Real} <: RadialFunction 
    β :: R = 3

    @assert β > 0 "The exponent `β` must be positive."
    @assert β % 2 != 0 "The exponent `β` must not be an even number."
end 

function ( φ :: Cubic )( ρ :: Real )
    (-1)^ceil(Int, φ.β/2 ) * ρ^φ.β
end

cpd_order( φ :: Cubic ) = ceil( Int, φ.β/2 )
df(φ :: Cubic, ρ :: Real ) = (-1)^(ceil(Int, φ.β/2)) * φ.β * ρ^(φ.β - 1)

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
@with_kw struct ThinPlateSpline <: RadialFunction
    k :: Int = 2

    @assert k > 0 && k % 1 == 0 "The parameter `k` must be a positive integer."
end

function (φ :: ThinPlateSpline )( ρ :: Real )
    (-1)^(k+1) * ρ^(2*k) * log( ρ )
end

cpd_order( φ :: ThinPlateSpline ) = φ.k + 1
df(φ :: ThinPlateSpline, ρ :: Real ) = ρ == 0 ? 0 : (-1)^(φ.k+1) * ρ^(2*φ.k - 1) * ( 2 * φ.k * log(ρ) + 1)

# !!! note 
#     The thin plate spline with `k = 1` is not differentiable at `ρ=0` but we define the derivative
#     as 0, which makes results in a continuous extension.