# quick script i use during development …

using RBFModels

f = x -> [ 1 + x[1]; sum(x.^2) ]
X = [ -3 .+ 6 * rand(2) for i = 1 : 5]
Y = f.(X)

# Initialize the `RadialFunction` to use for the RBF model:
φ = Multiquadric()

# Construct an interpolating model with linear polynomial tail:
rbf = RBFInterpolationModel( X, Y, φ, 1, static_arrays = false)