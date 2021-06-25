using RadialBasisFunctionModels
using Test

test_files = ["README.jl", "test_machine_wrapper.jl", "test_radial_funcs.jl", "test_rbf_models.jl"]

@testset "RadialBasisFunctionModels.jl" begin
    for fn in test_files
        include(fn)
    end
end
