using RadialBasisFunctionModels

using Test

f = x -> x^2

X = collect( LinRange(-4,4,5) )
Y = f.(X)

φ = Multiquadric()

rbf = RBFModel( X, Y, φ, 1)

Z = rbf.(X)

@test typeof(Z[1]) <: AbstractVector{<:Real}
@test length(Z[1]) == 1

@test all( isapprox(Z[i][1], Y[i]; atol = 1e-10) for i = 1 : length(Z) )

@test rbf( [ X[1], ] ) == Z[1]

rbf_scalar = RBFInterpolationModel( X, Y, φ, 1; vector_output = false)
Z_scalar = rbf_scalar.( X )

@test(
Z_scalar isa Vector{Float64}
)#jl
@test( #jl
all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) )
)

rbf_vec = RBFInterpolationModel( X, Y, φ, 1; static_arrays = false)

Xstatic = RadialBasisFunctionModels.SVector{1}(X[1])
@test(
rbf_vec(Xstatic) isa RadialBasisFunctionModels.SVector && rbf_vec(X[1]) isa Vector
)

X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 1; static_arrays = false )
@test(
rbf_f0.(X_f0) isa Vector{Vector{Float32}}
)#jl

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

