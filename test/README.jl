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

@test(#jl
Z_scalar isa Vector{Float64}
)#jl
@test(
all( Z_scalar[i] == Z[i][1] for i = 1 : length(Z) )
)

X_f0 = Float32.(X)
Y_f0 = f.(X_f0)
rbf_f0 = RBFInterpolationModel( X_f0, Y_f0, φ, 1)
@test(#jl
rbf_f0.(X_f0) isa Vector{Vector{Float32}}
)#jl

using StaticArrays
features = [ @SVector(rand(3)) for i = 1 : 5 ]
labels = [ @SVector(rand(3)) for i = 1 : 5 ]
centers = SVector{5}(features)
rbf_sized = RBFModel( features, labels; centers )

x_vec = rand(3)
x_s = SVector{3}(x_vec)
x_m = MVector{3}(x_vec)
x_sized = SizedVector{3}(x_vec)

@test(#jl
rbf_sized( x_vec ) isa Vector
)#jl
@test(#jl
rbf_sized( x_s ) isa SVector
)#jl
@test(#jl
rbf_sized( x_m ) isa MVector
)#jl
@test(#jl
rbf_sized( x_sized ) isa SizedVector
)#jl

#	:gaussian, :multiquadric, :inv_multiquadric, :cubic, :thin_plate_spline

RBFModel(features, labels, :gaussian)

RBFModel(features, labels, :mulitquadric, [1.0, 1//2])

using MLJBase
X,y = @load_boston

r = RBFInterpolator(; kernel_name = :multiquadric )
R = machine(r, X, y)

MLJBase.fit!(R)
MLJBase.predict(R, X)

X = [ rand(2) for i = 1 : 10 ]
Y = [ rand(2) for i = 1 : 10 ]

R = RBFMachine(features = X, labels = Y, kernel_name = :gaussian )
RadialBasisFunctionModels.fit!(R)
@test(#jl
R( X[1] ) ≈ Y[1]
)#jl

R = RBFMachine()
add_data!(R, X, Y)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

