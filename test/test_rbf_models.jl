using ForwardDiff: sqrt
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using RadialBasisFunctionModels
using Test 
using ForwardDiff
using RadialBasisFunctionModels.StaticArrays

#%%
@testset "1D-Data" begin 
	for T in subtypes(AbstractFloat)
		φ = Cubic()	# is good at interpolating ill-conditioned data
		features = rand(T, 5)
		labels = rand(T, 5)

		rbf = RBFModel( features, labels, φ; )

		@test typeof(rbf).parameters[1] == true
		@test rbf(features[1]) isa Vector{T}
		if T != Float16 
			for (f,l) in zip(features, labels) 
				@test isapprox(rbf(f)[1], l; atol = 0.01)
			end
		end
		@test rbf.rbf.kernels[1].c isa Vector{T}

		rbf = RBFModel( features, labels; vector_output = false )
		@test typeof(rbf).parameters[1] == false
		@test rbf(features[1]) isa T
		@test rbf.rbf.kernels[1].c isa Vector{T}
		
		rbf = RBFModel( features, labels; vector_output = false )
		@test typeof(rbf).parameters[1] == false
		@test rbf(features[1]) isa T
		@test rbf.rbf.kernels[1].c isa Vector{T}
	end
end

#%%

@testset "NonStaticData" begin 
	
	for T in subtypes(AbstractFloat)
		for φ in [Gaussian(), InverseMultiquadric(), Multiquadric(), Cubic(), ThinPlateSpline(1)]
			n_in = rand(1:10)
			n_out = rand(1:10)
			num_data = n_in + 1
			features = [ rand(T, n_in) for i = 1 : num_data ]
			labels = [ rand(T, n_out) for i = 1 : num_data ]

			rbf = RBFModel( features, labels, φ, 1 )

			T == Float16 || @test all( rbf( f ) ≈ l for (f,l) in zip(features,labels) )

			x = features[1]
			y = labels[1]

			@test rbf(x) isa Vector{T}

			for (x,y) in zip(features, labels)
				for ℓ = 1 : n_out
					G = grad(rbf, x, ℓ)
					aG = auto_grad(rbf, x, ℓ)
					@test G isa Vector{T}
					@test aG isa Vector{T}
					T == Float16 || @test grad( rbf, x, ℓ ) ≈ ForwardDiff.gradient( ξ -> rbf(ξ, ℓ), x )
					T == Float16 || @test G ≈ aG
				end
			end
		end
	end
end

#%%
@testset "StaticArraysAndTypes" begin 
		
	FTypes = Iterators.Stateful(Iterators.cycle(subtypes(AbstractFloat)))
	Rads = Iterators.Stateful(Iterators.cycle([Cubic(),Gaussian(),Multiquadric(),InverseMultiquadric(),ThinPlateSpline(1)])) 

	SizedTypes = [ SizedVector, MVector, SVector ]

	for FT in SizedTypes, LT in SizedTypes, CT in SizedTypes 
		n_in = rand(1:10)
		n_out = rand(1:10)
		num_data = n_in + 1
		T = popfirst!(FTypes)
		φ = popfirst!(Rads)

		sized_features = [ FT{n_in}(rand(T,n_in)) for i = 1 : num_data ]
		sized_labels = [ LT{n_out}(rand(T, n_out)) for i = 1 : num_data ]

		rbf = RBFModel(sized_features, sized_labels, φ)
		
		@test length( rbf.rbf.kernels ) == num_data
		@test rbf.rbf.kernels[1].c isa FT
		@test !(rbf.rbf.weights isa StaticArray)	# no sized array of sized centers was provided

		@test !(rbf.psum.weights isa StaticArray)	# no sized array of sized centers was provided

		# check if _type_guard works
		x = Vector(sized_features[1])
		@test rbf.rbf(x) isa Vector
		@test rbf.psum(x) isa Vector
		@test rbf(x) isa Vector
		for XT in SizedTypes
			ξ = XT{n_in}(x)
			@test rbf.rbf(ξ) isa XT 
			@test rbf.psum(ξ) isa XT 
			@test rbf(ξ) isa XT
		end
		## if we provide length information on the number of centers, we should get sized output
		sized_scenters = SizedVector{num_data}( sized_features )
		rbf = RBFModel( sized_features, sized_labels, φ; centers = sized_scenters )

		@test rbf.rbf.weights isa StaticArray
		@test rbf.psum.weights isa StaticArray
		
		@test rbf.rbf(x) isa Vector
		@test rbf.psum(x) isa Vector
		@test rbf(x) isa Vector
		for XT in SizedTypes
			ξ = XT{n_in}(x)
			@test rbf.rbf(ξ) isa XT 
			@test rbf.psum(ξ) isa XT 
			@test rbf(ξ) isa XT
		end
	end
end

#%%
Pkg.activate(current_env)
