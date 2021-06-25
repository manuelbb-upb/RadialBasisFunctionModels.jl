using Base: AbstractFloat
import MLJModelInterface
import Tables
const MMI = MLJModelInterface

mutable struct RBFInterpolator{F <: AbstractFloat} <: MMI.Deterministic
	kernel_name :: Symbol
	kernel_args :: Vector{F}
	polynomial_degree :: Int
end

_precision( :: RBFInterpolator{F} ) where F = F

function MMI.clean!( m :: RBFInterpolator ) :: String
	warning = "" 
	if m.kernel_name ∉ [:gaussian, :multiquadric, :inv_multiquadric, :cubic, :thin_plate_spline] 
		warning *= "Parameter value $(m.kernel_name) for `kernel_name` not supported, using `:gaussian`.\n"
		m.kernel_name = :gaussian
	end
	 
	φ_constructor = SymbolToRadialConstructor[ m.kernel_name ]
	try 
		φ_constructor( m.kernel_args... )
	catch MethodError
		warning *= "Invalid kernel arguments. Using defaults for $(m.kernel_name).\n"
		m.kernel_args = collect(getfield.(φ_constructor(), fn for fn in fieldnames(φ_constructor)))
	end

	φ = φ_constructor( m.kernel_args... )
	if m.polynomial_degree < cpd_order(φ) - 1
		warning *= "Need a polynomial of degree at least $(cpd_order(φ) - 1).\n"
		m.polynomial_degree = cpd_order(φ) - 1
	end
	
	warning
end

function RBFInterpolator(; kernel_name :: Symbol = :gaussian, kernel_args :: Vector{<:Real} = [1,], polynomial_degree = 1)
	if !(eltype(kernel_args) <: AbstractFloat)
		kernel_args = Float16.(kernel_args)
	end
	model = RBFInterpolator(kernel_name, kernel_args,polynomial_degree)
	message = MMI.clean!( model )
	isempty(message) || @warn message 
	return model
end


function MMI.fit( mod :: RBFInterpolator, verbosity, X :: AbstractVector{<:AbstractVector}, y :: AbstractVector )	
	φ_constructor = SymbolToRadialConstructor[ mod.kernel_name ]
	φ = φ_constructor( mod.kernel_args... )

	inner_model = RBFInterpolationModel( X, y, φ , mod.polynomial_degree)
	return inner_model, nothing, nothing
end

function MMI.reformat( ::RBFInterpolator, X,  y :: AbstractVector ) 
	features = [ collect(Real,v)  for v in collect( Tables.rows(X) ) ] 
	labels = Real.(y)
	return (features, labels)
end

function MMI.reformat( ::RBFInterpolator, X,  y ) 
	features = [ collect(Real,v)  for v in collect( Tables.rows(X) ) ] 
	labels =  [ collect(Real,v)  for v in collect( Tables.rows(y) ) ] 
	return (features, labels)
end

function MMI.reformat( ::RBFInterpolator, X ) 
	features = [ collect(Real,v)  for v in collect( Tables.rows(X) ) ] 
	return (features,)
end

function MMI.selectrows( ::RBFInterpolator, I, X :: AbstractVector{<:AbstractVector} )
	return (X[I],)
end

function MMI.selectrows( ::RBFInterpolator, I, X :: AbstractVector{<:AbstractVector}, y :: Union{AbstractVector, Vector{<:AbstractVector}} )
	return (X[I],y[I])
end

function MMI.predict( :: RBFInterpolator, inner_model :: RBFInterpolationModel, Xnew :: AbstractVector{<:AbstractVector})
	return inner_model.(Xnew)
end

function MMI.fitted_params(model::RBFInterpolator, inner_model) ::NamedTuple
	return (
		rbf_weights = inner_model.model.rbf.weights,
		rbf_centers = [ k.c for k ∈ inner_model.model.rbf.kernels ],
		polynomial_weights = inner_model.model.psum.weights
	)
end

MMI.metadata_pkg(
	RBFInterpolator,
	name="RBFInterpolator",
        package_uuid="48790e7e-73b2-491a-afa5-62818081adcb",
        package_url="https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl",
	package_license="MIT",
        is_pure_julia=true,
	is_wrapper=true,	# TODO is it a wrapper? 
)

MMI.metadata_model(
    RBFInterpolator,
    input_scitype=MMI.Table(MMI.Continuous),
    target_scitype=Union{AbstractVector{MMI.Continuous}, MMI.Table(MMI.Continuous)},
    supports_weights=true,
    docstring="Radial Basis Function Interpolator",
    load_path="RadialBasisFunctionModels.RBFInterpolator"
)