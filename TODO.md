# Tests

## Construction 
* Do the different constructors work as expected? 
  - `RBFModel` 
  - `RBFInterpolationModel`
* Are the keyword arguments respected?
* All training data can be passed as Vector-of-Vectors or Vector-of-SVectors or SVector-of-SVectors.
* In 1DnD or nD1D situations, (S)Vector-of-numbers should work too.

## Evaluation
* Does an `RBFInterpolationModel` (approximately) interpolate all training data?
* Is evaluation type stable? 
* Put in `x :: Vector{Float64}`, return vector of same type.
* By default, all `RBFModel`s should take and return vectors.
  If a model is constructed with `vector_output = false` AND if the training labels are 1D, then output is scalar.
  If `vector_output = false` but training labels are at least 2D then vectors should be returned.
* For `rbf::RBFModel` evaluating a single output, i.e., `rbf(x, 1)` should return a scalar.

## Performance (Maybe in Docs? -> Results are computed and shown)
* How is construction speed and evaluation speed and differentiation speed affected by different data format.
  - When the constructor is provided a Vector-of-Vectors or Vector-of-SVectors or SVector-of-SVectors.
  - When we evaluate or differentiate passing a `Vector{<:Real}` vs `SVector{<:Real}`
* `auto_grad`/`auto_jac` vs `grad`, `jac`.

Test for different data dimensions!

## Derivatives

* Do `auto_grad` and `grad` produce the same results?
* Do `auto_jac` and `jac` produce the same results?
* Do the gradients and the jacobian rows correspond?
* Do `eval_and_jac` and `eval_and_grad` return the same thing as evaluating and calling `jac` or `grad` respectively?
* Compare with finite differencing methods.
* Are the methods type-stable? Put in `x :: Vector{Float64}`, return gradient of same type, or jacobian of type `Matrix{Float64}`.
  If `x :: SVector` -> result should be an StaticArray, too.

Test for different `RadialFunction`s (Gaussian, Multiquadric etc.) and different Polynomial degrees (-1, 0, 1).

# Implement

* Warning if kwarg `poly_deg` is changed by constructor.
* Hessians?