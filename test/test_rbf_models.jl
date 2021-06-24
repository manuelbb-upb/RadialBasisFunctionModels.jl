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

		rbf = RBFModel( features, labels, φ )

		@test typeof(rbf).parameters[1] == true
		@test rbf(features[1]) isa Vector{T}
		#if T != Float16 
			for (f,l) in zip(features, labels) 
				@test isapprox(rbf(f)[1], l; atol = 0.01)
			end
		#end
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

@testset "Vector{Vector{AbstractFloat}}" begin 
	n_in = 2
	n_out = 3
	for T in subtypes(AbstractFloat)
		for φ in [Gaussian(), InverseMultiquadric(), Multiquadric(), Cubic(), ThinPlateSpline(1)]

			features = [ rand(T, n_in) for i = 1 : 5 ]
			labels = [ rand(T, n_out) for i = 1 : 5 ]

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
@testset "StaticArrays" begin 
	n_in = 2 
	n_out = 3
	φ = Gaussian()
	T = Float32

	sized_features = [ SizedVector{n_in}(rand(T,n_in)) for i = 1 : 5 ]
	sized_labels = [ SizedVector{n_out}(rand(T, n_out)) for i = 1 : 5 ]

	rbf = RBFModel(sized_features, sized_labels, φ)
	xs = sized_features[1]
	x = Vector(xs)
	
	@test length( rbf.rbf.kernels ) == 5
	@test rbf.rbf.kernels[1].c isa SizedVector
	@test !(rbf.rbf.weights isa StaticArray)	# no sized array of sized centers was provided
	@test rbf.rbf(xs) isa Vector{T}	# hence, overall output not sized neither
	@test rbf.rbf(x) isa Vector{T}

	@test !(rbf.psum.weights isa StaticArray)	# no sized array of sized centers was provided
	@test rbf.psum(xs) isa Vector{T}	# hence, overall output not sized neither
	@test rbf.psum(x) isa Vector{T}

	@test rbf(xs) isa Vector{T}
	@test rbf(x) isa Vector{T}

	## if we provide length information on the number of centers, we should get sized output
	sized_scenters = SizedVector{5}( sized_features )
	rbf = RBFModel( sized_features, sized_labels, φ; centers = sized_scenters )

	@test rbf.rbf.weights isa StaticArray
	@test rbf.psum.weights isa StaticArray
	@test rbf.rbf(xs) isa SizedVector 
	@test rbf.psum(xs) isa SizedVector
	@test rbf(xs) isa SizedVector
	@test rbf.rbf(x) isa Vector 
	@test rbf.psum(x) isa Vector
	@test rbf(x) isa Vector
end

#%%
Pkg.activate(current_env)
