using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using RadialBasisFunctionModels
using Test
using InteractiveUtils: subtypes 

#%% initialize empty machine 
# should default to 64 bit precision
@testset "Initialization" begin 
	mach = RBFMachine(;poly_deg = 1)
	@test mach.features isa Vector{Vector{Float64}}
	@test mach.labels isa Vector{Vector{Float64}}
	@test isnothing(mach.kernel_args)
	@test mach.kernel_name == :gaussian

	@test_throws AssertionError RadialBasisFunctionModels.fit!(mach)

	add_data!( mach, rand(1), rand(1) )
	@test_throws AssertionError RadialBasisFunctionModels.fit!(mach)  

	add_data!( mach, rand(1), rand(1) )
	@test isnothing(RadialBasisFunctionModels.fit!(mach))

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

		RadialBasisFunctionModels.fit!(mach)

		@test mach.valid
		T == Float16 || @test mach( mach.features[1] ) ≈ mach.labels[1]
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

			RadialBasisFunctionModels.fit!(mach)

			@test mach.valid
			# @test mach( mach.features[1] ) ≈ mach.labels[1] fails for Float16
			@test mach( features[1] ) isa Vector{T}
		end
	end
end

#%%
@testset "Kernels" begin
	features = [ rand(2) for i = 1 : 4 ]
	labels = [ rand(3) for i = 1 : 4 ]
	for kn in keys(RadialBasisFunctionModels.SymbolToRadialConstructor)
		if kn != :thin_plate_spline # default thin plate spline has cpd order = 3 => needs quadratic polynomial
			mach = RBFMachine(; 
				kernel_name = kn,
				poly_deg = 1,
				features, labels 
			)
			@test isnothing(RadialBasisFunctionModels.fit!(mach))
			@test mach.valid 
			@test all( mach(features[i]) ≈ labels[i] for i = 1 : 4 )
		else
			@test_throws AssertionError RBFMachine(; 
				kernel_name = kn,
				poly_deg = 1,
				features, labels 
			)

			mach = RBFMachine(; 
				kernel_name = kn,
				kernel_args = [1,],
				poly_deg = 1,
				features, labels 
			)
			@test isnothing(RadialBasisFunctionModels.fit!(mach))
			@test mach.valid 
			@test all( mach(features[i]) ≈ labels[i] for i = 1 : 4 )
		end 
	end

	# Multiquadric has cpd order 1 => needs constant polynomial
	@test_throws AssertionError RBFMachine( kernel_name = :multiquadric, poly_deg = -1 )
	
	# Cubic has order 2
	@test_throws AssertionError RBFMachine( kernel_name = :cubic, poly_deg = -1 )
	@test_throws AssertionError RBFMachine( kernel_name = :cubic, poly_deg = 0 )

	for kernel_name in [:gaussian, :inv_multiquadric], poly_deg in [-1,0,1]
		@test RBFMachine(;kernel_name, poly_deg) isa RBFMachine
	end
end

#%%
Pkg.activate(current_env)
