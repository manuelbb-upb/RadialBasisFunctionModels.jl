# RadialBasisFunctionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/dev)
[![Build Status](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/actions)
[![Coverage](https://codecov.io/gh/manuelbb-upb/RadialBasisFunctionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/RadialBasisFunctionModels.jl)

# Description
This package provides Radial Basis Function (RBF) models with polynomial tails.
RBF models are a special case of kernel machines that can interpolate high-dimensional
and nonlinear data.

# Usage Examples

First load the `RadialBasisFunctionModels` package.

```julia
using RadialBasisFunctionModels
```

## Interpolating RBF Model

### One dimensional data
The main type `RBFModel` uses vectors internally, but we can easily
interpolate 1-dimensional data.
Assume, e.g., we want to interpolate ``f:ℝ → ℝ, f(x) = x^2``:

```julia
f = x -> x^2
```

Define 5 training sites `X` and evaluate to get `Y`

```julia
X = collect( LinRange(-4,4,5) )
Y = f.(X)
```

Initialize the `RadialFunction` to use for the RBF model:

```julia
φ = Multiquadric()
```

Construct an interpolating model with linear polynomial tail:

```julia
rbf = RBFModel( X, Y, φ, 1)
```

We can evaluate `rbf` at the data points;
By default, vectors are returned.

```julia
Z = rbf.(X)
```

Now

```julia
Z isa Vector
```

and

```julia
Z[1] isa AbstractVector{<:Real}
```

The results should be close to the data labels `Y`, i.e., `Z[1] ≈ Y[1]` etc.

`X` contains Floats, but we can pass them to `rbf`.
Usually you have feature vectors and they are always supported:

```julia
@test rbf( [ X[1], ] ) == Z[1]
```

For 1 dimensional labels we can actually disable the vector output:

```julia
rbf_scalar = RBFInterpolationModel( X, Y, φ, 1; vector_output = false)
Z_scalar = rbf_scalar.( X )

Z_scalar isa Vector{Float64}
all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) )
```

The data precision of the training data is preserved when determining the model
coefficients.
Accordingly, the return type precision is also at least that of the training data.

```julia
X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 1)
rbf_f0.(X_f0) isa Vector{Vector{Float32}}
```

If you are using statically sized arrays, they work too!
You can provide a vector of statically sized arrays or, if
you have only few centers ( number of variables × number of centers <= 100),
provide a statically sized vector of statically sized vectors to maybe profit
from faster matrix multiplications when evaluating:

```julia
using StaticArrays
features = [ @SVector(rand(3)) for i = 1 : 5 ]
labels = [ @SVector(rand(3)) for i = 1 : 5 ]
centers = SVector{5}(features)
rbf_sized = RBFModel( features, labels; centers )
```

Now, the model uses sized matrices internally.
For most input vectors a `SizedVector` would be returned.
But there is a "type guarding" function for static arrays so that output has the same
array type (by conversion, if necessary):

```julia
x_vec = rand(3)
x_s = SVector{3}(x_vec)
x_m = MVector{3}(x_vec)
x_sized = SizedVector{3}(x_vec)

rbf_sized( x_vec ) isa Vector
rbf_sized( x_s ) isa SVector
rbf_sized( x_m ) isa MVector
rbf_sized( x_sized ) isa SizedVector
```

### Using Kernel Names

Instead of initializing `RadialFunction`s beforehand,
their names can be used. Currently supported are:
```
:gaussian, :multiquadric, :inv_multiquadric, :cubic, :thin_plate_spline
```

You can do

```julia
RBFModel(features, labels, :gaussian)
```

or, to specify kernel arguments:

```julia
RBFModel(features, labels, :multiquadric, [1.0, 1//2])
```

## Machines

There is an MLJ wrapper for the RBFInterpolationModel, exported as `RBFInterpolator`.
It can be used like other regressors and takes the kernel name as a symbol (and kernel arguments as a vector).

```julia
using MLJBase
X,y = @load_boston

r = RBFInterpolator(; kernel_name = :multiquadric )
R = machine(r, X, y)

MLJBase.fit!(R)
MLJBase.predict(R, X)
```

You can do similar things (for vector valued data) with the `RBFMachineWithKernel`:

```julia
X = [ rand(2) for i = 1 : 10 ]
Y = [ rand(2) for i = 1 : 10 ]

R = RBFMachine(;features = X, labels = Y, kernel_name = :gaussian )
R isa RBFMachineWithKernel
RadialBasisFunctionModels.fit!(R)
R( X[1] ) ≈ Y[1]
```

Such a machine can be initialized empty and data can be added:

```julia
R = RBFMachine()
add_data!(R, X, Y)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

