# # RBFModels
# 
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/RBFModels.jl/stable)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/RBFModels.jl/dev)
# [![Build Status](https://github.com/manuelbb-upb/RBFModels.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/RBFModels.jl/actions)
# [![Coverage](https://codecov.io/gh/manuelbb-upb/RBFModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/RBFModels.jl)
# 

# # Description
# This package provides Radial Basis Function (RBF) models with polynomial tails.
# RBF models are a special case of kernel machines can interpolate high-dimensional 
# and nonlinear data.

# # Usage Examples 

# First load the `RBFModels` package.
using RBFModels

# We also use `Test` to validate the results and `BenchmarkTools` for comparisons.
using Test
using BenchmarkTools
# ## Interpolating RBF Model 

# ### One dimensional data
# The main type `RBFModel` uses vectors internally, but we can easily 
# interpolate 1-dimensional data.
# Assume, e.g., we want to interpolate ``f:ℝ → ℝ, f(x) = x^2``:
f = x -> x^2
# Define 5 training sites `X` and evaluate to get `Y`
X = collect( LinRange(-4,4,5) )
Y = f.(X)

# Initialize the `RadialFunction` to use for the RBF model:
φ = Multiquadric()

# Construct an interpolating model with linear polynomial tail:
rbf = RBFInterpolationModel( X, Y, φ, 1)

# We can evaluate `rbf` at the data points; 
# By default, vectors are returned and for small dimensions 
# `StaticArrays` are used.
Z = rbf.(X)
@test Z isa Vector{RBFModels.SVector{1,Float64}}
@test length(Z[1]) == 1
# The results should be close to the data labels `Y`.
@test all( isapprox(Z[i][1], Y[i]; atol = 1e-10) for i = 1 : length(Z) )

# `X` contains Floats, but we can pass them to `rbf`.
# Usually you have feature vectors and they are always supported:
@test rbf( [ X[1], ] ) == Z[1]

# For 1 dimensional labels we can actually disable the vector output:
rbf_scalar = RBFInterpolationModel( X, Y, φ, 1; vector_output = false)
Z_scalar = rbf_scalar.( X )
@test Z_scalar isa Vector{Float64}
@test all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) )

# Also, the `StaticArrays` can be disabled:
rbf_vec = RBFInterpolationModel( X, Y, φ, 0; static_arrays = false)
Z_vec = rbf_vec.(X)
@test Z_vec isa Vector{Vector{Float64}}

# Whether `StaticArrays` are used and if vectors are returned is 
# indicated by the type flags:
@test rbf isa RBFModel{true, true}          # SVectors and vector output
@test rbf_scalar isa RBFModel{true, false}  # SVectors and scalar output
@test rbf_vec isa RBFModel{false, true}     # Vectors and vector output

# The data precision of the training data is preserved when evaluating.
X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 0; static_arrays = false ) 
@test rbf_f0.(X_f0) isa Vector{Vector{Float32}}