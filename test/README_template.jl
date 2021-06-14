# # RadialBasisFunctionModels
# 
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/stable)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/dev)
# [![Build Status](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/actions)
# [![Coverage](https://codecov.io/gh/manuelbb-upb/RadialBasisFunctionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/RadialBasisFunctionModels.jl)
# 

# # Description
# This package provides Radial Basis Function (RBF) models with polynomial tails.
# RBF models are a special case of kernel machines can interpolate high-dimensional 
# and nonlinear data.

# # Usage Examples 

# First load the `RadialBasisFunctionModels` package.
using RadialBasisFunctionModels

# We also use `Test` to validate the results and `BenchmarkTools` for comparisons. #jl
using Test #md
using BenchmarkTools #md

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
rbf = RBFModel( X, Y, φ, 1)

# We can evaluate `rbf` at the data points; 
# By default, vectors are returned.
Z = rbf.(X)
# Now 
Z isa Vector #md
# and 
Z[1] isa AbstractVector{<:Real} #md

@test typeof(Z[1]) <: AbstractVector{<:Real} #jl
@test length(Z[1]) == 1     #jl
# The results should be close to the data labels `Y`, i.e., `Z[1] ≈ Y[1]` etc. 
@test all( isapprox(Z[i][1], Y[i]; atol = 1e-10) for i = 1 : length(Z) ) #jl 

# `X` contains Floats, but we can pass them to `rbf`.
# Usually you have feature vectors and they are always supported:
@test rbf( [ X[1], ] ) == Z[1]

# For 1 dimensional labels we can actually disable the vector output:
rbf_scalar = RBFInterpolationModel( X, Y, φ, 1; vector_output = false)
Z_scalar = rbf_scalar.( X )
typeof(Z_scalar[1])
@test Z_scalar isa Vector{Float64}  #jl
@test all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) ) #jl

# Also, the internal use of `StaticArrays` can be disabled:
rbf_vec = RBFInterpolationModel( X, Y, φ, 1; static_arrays = false)
@test Z_vec isa Vector{Vector{Float64}} #jl

# The return type of the evaluation function should be independent of that setting.
# It rather depends on the input type.
Xstatic = SVector{1}(X[1])
typeof(rbf_vec(Xstatic)) #md
@test rbf_vec(Xstatic) isa SVector  #jl

# The data precision of the training data is preserved when evaluating.
X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 1; static_arrays = false ) 
@test rbf_f0.(X_f0) isa Vector{Vector{Float32}} #jl

# Benchmarks for the small 1in1out data set. Construction:
creation_times = [ #md
    median(@benchmark( RBFInterpolationModel( X, Y, φ, 1))), #md
    median(@benchmark( RBFInterpolationModel( X, Y, φ, 1; vector_output = false))), #md
    median(@benchmark( RBFInterpolationModel( X, Y, φ, 1; static_arrays = false))), #md
    median(@benchmark( RBFInterpolationModel( X_f0, Y_f0, φ, 1))) #md
] #md
# Evaluation:
eval_times = [ #md
    median( @benchmark( rbf.(X) ) ), #md
    median( @benchmark( rbf_scalar.(X) ) ), #md
    median( @benchmark( rbf_vec.(X) ) ), #md
    median( @benchmark( rbf_f0.(X_f0) ) ) #md
] #md