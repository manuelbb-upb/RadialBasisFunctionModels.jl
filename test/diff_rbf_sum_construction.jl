import RadialBasisFunctionModels 
import Flux.Zygote as Zyg
using Test

#%%
function build_rbf_sum(num_kernels = 5, num_vars = 2, num_outputs = 3; static = false)
    φ = RadialBasisFunctionModels.Gaussian()
    centers = [ rand(num_vars) for i = 1 : num_kernels ]
    kernels = RadialBasisFunctionModels.make_kernels(φ, centers)

    weights = rand(num_kernels, num_outputs )
    RadialBasisFunctionModels.get_RBFSum( kernels, weights, num_vars, num_kernels, num_outputs; static_arrays = static)
end

l = function(x)
    rbf = build_rbf_sum(; static = true)
    sum( abs.(rbf(x)) )
end

@test !isnothing(Zyg.gradient( l, rand(2) )[1])

#%%
l = function(x)
    RadialBasisFunctionModels.canonical_basis(3, 2)
    x
end