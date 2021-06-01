# # Some Radial Functions 

# The **Gaussian** is defined by ``φ(ρ) = \exp \left( - (αρ)^2 \right)``, where 
# ``α`` is a shape parameter to fine-tune the function.

"""
    Gaussian( α = 1 ) <: RadialFunction

A `RadialFunction` with 
```math 
    φ(ρ) = \\exp( - (α ρ)^2 ).
```
"""
struct Gaussian{R<:Real} <: RadialFunction 
    α :: R

    function Gaussian( α :: R = 1 ) where R<:Real
        @assert α > 0 "The shape parameter `α` must be positive."
        return new{R}(α)
    end
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
struct Multiquadric{R<:Real,S<:Real} <: RadialFunction
    α :: R   # shape parameter 
    β :: S   # exponent 

    function Multiquadric(α :: R = 1, β :: S = 1//2 ) where {R<:Real, S<:Real}
        @assert α > 0 "The shape parameter `α` must be positive."
        @assert β % 1 != 0 "The exponent must not be an integer."
        @assert β > 0 "The exponent must be positive."
        new{R,S}(α,β)
    end
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
struct InverseMultiquadric{R<:Real,S<:Real} <: RadialFunction
    α :: R 
    β :: S 

    function InverseMultiquadric( α :: Real = 1, β :: Real = 1//2 ) where {R<:Real, S<:Real}
        @assert α > 0 "The shape parameter `α` must be positive."
        @assert β > 0 "The exponent must be positive."
        new{R,S}(α, β)
    end
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
struct Cubic{R<:Real} <: RadialFunction 
    β :: R

    function Cubic( β :: R = 3 ) where R<:Real
        @assert β > 0 "The exponent `β` must be positive."
        @assert β % 2 != 0 "The exponent `β` must not be an even number."
        new{R}(β)
    end
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

# !!! note 
#     The thin plate spline with `k = 1` is not differentiable at `ρ=0` but we define the derivative
#     as 0, which makes results in a continuous extension.