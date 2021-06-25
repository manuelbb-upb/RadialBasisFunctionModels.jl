using RadialBasisFunctionModels
using MLJBase

r = RadialBasisFunctionModels.RBFInterpolator(; kernel_name = :multiquadric, kernel_args = [1,])
X,y = @load_boston

R = machine(r, X, y)
fit!(R)

predict(R, X)