using Base: Float64
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using RadialBasisFunctionModels
using Test 
import ForwardDiff

#%%
@testset "Gaussian" begin
	φ = Gaussian()
	@test φ(0) == 1
	@test φ(0) isa Float64	# stanard precision of `exp`
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16

	φ = Gaussian( rand(Float16) )
	@test RadialBasisFunctionModels.cpd_order(φ) == 0
	@test φ(0) == 1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16
	
	φ = Gaussian( rand(Float32) )
	@test φ(0) == 1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float32
	
	φ = Gaussian( rand(Float64) )
	@test φ(0) == 1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float64
	@test φ(Float16(0)) isa Float64

	φ = Gaussian() 

	@test RadialBasisFunctionModels.df( φ, 0 ) == 0
	x = rand()
	@test RadialBasisFunctionModels.df( φ, x ) ≈ ForwardDiff.derivative( φ, x)
end

#%%
@testset "Multiquadric" begin
	φ = Multiquadric()
	@test RadialBasisFunctionModels.cpd_order(φ) == 1
	@test φ(0) == -1
	@test φ(0) isa Float64	# stanard precision of `sqrt`
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16

	φ = Multiquadric(; α = one(Float16) )
	@test φ(0) == -1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16
	
	φ = Multiquadric(; α = one(Float32) )
	@test φ(0) == -1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float32
	
	φ = Multiquadric(; α = one(Float64) )
	@test φ(0) == -1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float64
	@test φ(Float16(0)) isa Float64

	φ = Multiquadric() 
	x = rand()
	@test RadialBasisFunctionModels.df( φ, x ) ≈ ForwardDiff.derivative( φ, x)	
end

#%%
@testset "InverseMultiquadric" begin
	φ = InverseMultiquadric()
	@test RadialBasisFunctionModels.cpd_order(φ) == 0
	@test φ(0) == 1
	@test φ(0) isa Float64	# stanard precision of `sqrt`
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16

	φ = InverseMultiquadric(; α = one(Float16) )
	@test φ(0) == 1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16
	
	φ = InverseMultiquadric(; α = one(Float32) )
	@test φ(0) == 1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float32
	
	φ = InverseMultiquadric(; α = one(Float64) )
	@test φ(0) == 1
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float64
	@test φ(Float16(0)) isa Float64

	φ = InverseMultiquadric() 
	x = rand()
	@test RadialBasisFunctionModels.df( φ, x ) ≈ ForwardDiff.derivative( φ, x)	
end

#%%
@testset "Cubic" begin
	φ = Cubic()
	@test RadialBasisFunctionModels.cpd_order(φ) == 2
	@test φ(0) == 0
	@test φ(1) == 1
	@test φ(0) isa Int
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16

	φ = Cubic( Float64(3) )
	@test φ.β isa Int

	φ = Cubic() 

	@test RadialBasisFunctionModels.df( φ, 0 ) == 0
	x = rand()
	@test RadialBasisFunctionModels.df( φ, x ) ≈ ForwardDiff.derivative( φ, x)
end

#%%
@testset "ThinPlateSpline" begin
	φ = ThinPlateSpline()
	@test RadialBasisFunctionModels.cpd_order(φ) == 3
	@test φ(0) == 0
	@test φ(0.0) isa Float64
	@test φ(0f0) isa Float32 
	@test φ(Float16(0)) isa Float16

	@test RadialBasisFunctionModels.df( φ, 0 ) == 0
	x = rand()
	@test RadialBasisFunctionModels.df( φ, x ) ≈ ForwardDiff.derivative( φ, x)
end

#%%
Pkg.activate(current_env)