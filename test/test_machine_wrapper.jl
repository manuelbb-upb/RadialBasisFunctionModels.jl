using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using RadialBasisFunctionModels
using Test 

#%% initialize empty machine 
# should default to 64 bit precision
@testset "Initialization" begin 
	mach = RBFMachine(;poly_deg = 1)
	@test mach.features isa Vector{Vector{Float64}}
	@test mach.labels isa Vector{Vector{Float64}}
	@test isnothing(mach.kernel_args)
	@test mach.kernel_name == :gaussian

	@test_throws AssertionError fit!(mach)

	add_data!( mach, rand(1), rand(1) )
	@test_throws AssertionError fit!(mach)  

	add_data!( mach, rand(1), rand(1) )
	@test isnothing(fit!(mach))

	# do we interpolate?
	@test mach( mach.features[1] ) ≈ mach.labels[1]

	@test mach.valid

	add_data!(mach, rand(1), rand(1))

	@test !mach.valid
end

#%%
@testset "Precision" begin
	for T in subtypes(AbstractFloat)
		
		features = [ rand(T, 2) for i = 1 : 4 ]
		labels = [ rand(T, 3) for i = 1 : 4 ]

		mach = RBFMachine(; features, labels )
		F = typeof(mach)
		@test F.parameters[1] == typeof(features) == Vector{Vector{T}}
		@test F.parameters[2] == typeof(labels) == Vector{Vector{T}}

		fit!(mach)

		@test mach.valid
		@test mach( mach.features[1] ) ≈ mach.labels[1]
		@test mach( features[1] ) isa Vector{T}

		for S in subtypes(AbstractFloat)
			add_data!(mach, rand(S,2), rand(S,3) )
		end
		@test !mach.valid 
		@test mach.features isa Vector{Vector{T}}
		@test mach.labels isa Vector{Vector{T}}
		
	end
end

#%% Test if precision is kept if kernel_args have different precision
@testset "PrecisionII" begin
	for T in subtypes(AbstractFloat)
		for S in subtypes(AbstractFloat)
			features = [ rand(T, 2) for i = 1 : 4 ]
			labels = [ rand(T, 3) for i = 1 : 4 ]

			mach = RBFMachine(; 
				kernel_name = :gaussian,
				kernel_args = ones( S, 1 ),
				features, labels 
			)
			F = typeof(mach)
			@test F.parameters[1] == typeof(features) == Vector{Vector{T}}
			@test F.parameters[2] == typeof(labels) == Vector{Vector{T}}

			fit!(mach)

			@test mach.valid
			# @test mach( mach.features[1] ) ≈ mach.labels[1] fails for Float16
			@test mach( features[1] ) isa Vector{T}
		end
	end
end

#%%
Pkg.activate(current_env)
