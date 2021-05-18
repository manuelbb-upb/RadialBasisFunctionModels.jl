# quick script i use during development …
using StaticArrays
using RBFModels
import Flux.Zygote as Zyg

f = x -> [ 1 + x[1]; sum(x.^2) ]
X = [ -3 .+ 6 * rand(2) for i = 1 : 5]
Y = f.(X)

Ys = [ SVector{2}(y) for y ∈ Y ]
Xs = [ SVector{2}(x) for x ∈ X ]

# Initialize the `RadialFunction` to use for the RBF model:
φ = Gaussian()

# Construct an interpolating model with linear polynomial tail:
rbf = RBFModel( X, Y, φ, -1)

#=
using StaticArrays
ξ = @SVector(rand(2))
jac( rbf, ξ )
auto_jac( rbf, ξ)
=#
#%%

l = function( x )
    global X,Y,φ
    rbf_m = RBFModel( X, Y, φ, -1; static_arrays = false )
    return sum( abs.(rbf_m(x)) )
end

#Zyg.gradient( l, rand(2))
    