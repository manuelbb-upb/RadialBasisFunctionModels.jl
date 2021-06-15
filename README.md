# RadialBasisFunctionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/dev)
[![Build Status](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/actions)
[![Coverage](https://codecov.io/gh/manuelbb-upb/RadialBasisFunctionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/RadialBasisFunctionModels.jl)

# Description
This package provides Radial Basis Function (RBF) models with polynomial tails.
RBF models are a special case of kernel machines can interpolate high-dimensional
and nonlinear data.

# Usage Examples

First load the `RadialBasisFunctionModels` package.

````julia
using RadialBasisFunctionModels
````

## Interpolating RBF Model

### One dimensional data
The main type `RBFModel` uses vectors internally, but we can easily
interpolate 1-dimensional data.
Assume, e.g., we want to interpolate ``f:ℝ → ℝ, f(x) = x^2``:

````julia
f = x -> x^2
````

Define 5 training sites `X` and evaluate to get `Y`

````julia
X = collect( LinRange(-4,4,5) )
Y = f.(X)
````

Initialize the `RadialFunction` to use for the RBF model:

````julia
φ = Multiquadric()
````

Construct an interpolating model with linear polynomial tail:

````julia
rbf = RBFModel( X, Y, φ, 1)
````

We can evaluate `rbf` at the data points;
By default, vectors are returned.

````julia
Z = rbf.(X)
````

Now

````julia
Z isa Vector
````

and

````julia
Z[1] isa AbstractVector{<:Real}
````

The results should be close to the data labels `Y`, i.e., `Z[1] ≈ Y[1]` etc.

`X` contains Floats, but we can pass them to `rbf`.
Usually you have feature vectors and they are always supported:

````julia
@test rbf( [ X[1], ] ) == Z[1]
````

For 1 dimensional labels we can actually disable the vector output:

````julia
rbf_scalar = RBFInterpolationModel( X, Y, φ, 1; vector_output = false)
Z_scalar = rbf_scalar.( X )

Z_scalar isa Vector{Float64}
@test( #jl
all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) )
````

Also, the internal use of `StaticArrays` can be disabled:

````julia
rbf_vec = RBFInterpolationModel( X, Y, φ, 1; static_arrays = false)
````

The return type of the evaluation function should be independent of that setting.
It rather depends on the input type.

````julia
Xstatic = RadialBasisFunctionModels.SVector{1}(X[1])
rbf_vec(Xstatic) isa RadialBasisFunctionModels.SVector && rbf_vec(X[1]) isa Vector
````

The data precision of the training data is preserved when evaluating.

````julia
X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 1; static_arrays = false )
rbf_f0.(X_f0) isa Vector{Vector{Float32}}
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

