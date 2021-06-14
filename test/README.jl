using RadialBasisFunctionModels

f = x -> x^2

X = collect( LinRange(-4,4,5) )
Y = f.(X)

φ = Multiquadric()

rbf = RBFModel( X, Y, φ, 1)

Z = rbf.(X)

@test typeof(Z[1]) <: AbstractVector{<:Real}
@test length(Z[1]) == 1

@test all( isapprox(Z[i][1], Y[i]; atol = 1e-10) for i = 1 : length(Z) ) #jl

@test rbf( [ X[1], ] ) == Z[1]

rbf_scalar = RBFInterpolationModel( X, Y, φ, 1; vector_output = false)
Z_scalar = rbf_scalar.( X )
typeof(Z_scalar[1])
@test Z_scalar isa Vector{Float64}
@test all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) )

rbf_vec = RBFInterpolationModel( X, Y, φ, 1; static_arrays = false)
@test Z_vec isa Vector{Vector{Float64}}

Xstatic = SVector{1}(X[1])
@test rbf_vec(Xstatic) isa SVector

X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 1; static_arrays = false )
@test rbf_f0.(X_f0) isa Vector{Vector{Float32}}

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

