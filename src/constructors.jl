
# This file is included from within RBFModels.jl #src 

# ## Constructors
# ###  Polynomial Basis 

# Any constructor of an `RBFModel` must solve for the coefficients.
# To build the equation system, we need a basis ``\{p_j\}_{1 \le j \le Q}`` of ``Π_d(ℝ^n)``.

# The canonical basis is ``x_1^{α_1} x_2^{α_2} … x_n^{α_n}`` with 
# ``α_i ≥ 0`` and ``Σ_i α_i ≤ d``.
# For ``\bar{d} \le d`` we can recursively get the non-negative integer solutions for 
# ``Σ_i α_i = \bar{d}`` with the following function:

@doc """
    non_negative_solutions( d :: Int, n :: Int)

Return array of solution vectors ``[x_1, …, x_n]`` to the equation
``x_1 + … + x_n = d``
where the variables are non-negative integers.
"""
function non_negative_solutions( d :: Int, n :: Int )
    if n == 1
        return d
    else
        solutions = [];
        for i = 0 : d
            ## make RHS smaller by and find all solutions of length `n-1`
            ## then concatenate with difference `d-i`
            for shorter_solution ∈ non_negative_solutions( i, n - 1)
                push!( solutions, [ d-i ; shorter_solution ] )
            end
        end
        return solutions
    end
end

# We use `DynamicPolynomials.jl` to generate the Polyomials.
# Furthermore, we employ Memoization (via `Memoize.jl` and `ThreadSafeDicts`)
# to save the result for successive usage.
@doc """
    canonical_basis( n:: Int, d :: Int )

Return the canonical basis of the space of `n`-variate 
polynomials of degree at most `d`.
"""
@memoize ThreadSafeDict function canonical_basis( n :: Int, d :: Int )
    DP.@polyvar Xvar[1 : n]
    basis = DP.Polynomial{true,Int}[] # list of basis polynomials
    for d̄ = 0 : d 
        for multi_exponent ∈ non_negative_solutions( d̄, n )
            push!( basis, DP.Polynomial(prod( Xvar .^ multi_exponent ) ))
        end
    end
    return basis
end

# ### Solving the Equation System 
# For now, we use the `\` operator to solve `A * coeff = RHS`.
# Furthermore, we allow for different interpolation `sites` and 
# RBF centers by allowing for passing `kernels`.
const VecOfVecs{T} = AnyVec{<:AnyVec{<:T}}

@doc """
    coefficients(sites, values, kernels, rad_funcs, polys )

Return the coefficient matrices `w` and `λ` for an rbf model 
``r(x) = Σ_{i=1}^N wᵢ φ(\\|x - x^i\\|) + Σ_{j=1}^M λᵢ pᵢ(x)``,
where ``N`` is the length of `rad_funcs` (and `centers`) and ``M``
is the length of `polys`.

The arguments are 
* an array of data sites `sites` with vector entries from ``ℝ^n``.
* an array of data values `values` with vector entries from ``ℝ^k``.
* an array of `ShiftedKernel`s.
* an array `polys` of polynomial basis functions.
"""
function coefficients( 
        sites :: VecOfVecs, 
        values :: VecOfVecs, 
        kernels :: AnyVec{ShiftedKernel},
        polys :: Vector{<:DynamicPolynomials.Polynomial} 
    )

    n_out = length(values[1])
    
    ## Φ-matrix, N columns =̂ basis funcs, rows =̂ sites
    N = length(kernels);
    Φ = hcat( (k.(sites) for k ∈ kernels)... )
    
    ## P-matrix, N × Q
    Q = length(polys)
    P = hcat( (p.(sites) for p ∈ polys)... )

    ## system matrix A
    Z = zeros( eltype(Φ), Q, Q )
    A = [ Φ  P;
          P' Z ];

    ## build rhs
    RHS = [
        transpose( hcat( values... ) );
        zeros( eltype(eltype(values)), Q, size(F,2) )
    ];

    ## solve system
    coeff = A \ RHS 

    @show typeof(coeff)

    ## return w and λ
    if isa( sites[1], StaticArray )
        return SMatrix{N,n_out}(coeff[1 : N, :]), SMatrix{Q, n_out}(coeff[N+1 : end, :])
    else
        return coeff[1 : N, :], coeff[N+1 : end, :]
    end
end

# ### The Actual, Usable Constructor 

# We want the user to be able to pass 1D data as scalars and use the following helpers:
const NumberOrVector = Union{<:Real, AnyVec{<:Real}}

function ensure_vec_of_vecs( before :: AnyVec{ <:Real }, :: Val{false} )
    [ [vec,] for vec ∈ before ]
end
function ensure_vec_of_vecs( before :: AnyVec{ <:Real }, :: Val{true} )
    [ SVector{1}([vec,]) for vec ∈ before ]
end
ensure_vec_of_vecs( before :: AnyVec{ <:AnyVec }, args...) = before


# Helpers to create kernel functions. Should return `SVector` when appropriate. 
"Return a list of `ShiftedKernel`s."
function _make_kernels( φ_arr :: Union{RadialFunction, AnyVec{RadialFunction}}, sites :: AnyVec )
    if φ_arr isa RadialFunction
        φ_arr = [φ_arr for i = 1:length(sites)]
    end
    return [ ShiftedKernel(φ_arr[i], sites[i]) for i = eachindex( φ_arr ) ]
end

"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `sites`."
make_kernels( φ_arr, centers :: AnyVec{<:Vector} ) = _make_kernels( φ_arr, centers )
make_kernels( φ_arr, centers :: AnyVec{<:StatVec} ) = SVector{length(sites)}(_make_kernels(φ_arr,centers))

# We now have all ingredients for the basic outer constructor:

"""
    RBFModel( features, labels, φ = Multiquadric(), poly_deg = 1; kwargs ... )

Construct a `RBFModel` from the feature vectors in `features` and 
the corresponding labels in `lables`, where `φ` is a `RadialFunction` or a vector of 
`RadialFunction`s.\
Scalar data can be used, it is transformed internally. \
StaticArrays can be used, e.g., `features :: Vector{<:SVector}`. 
Do so for all data to benefit from possible speed-ups.\
If the degree of the polynomial tail, `poly_deg`, is too small it will be set to `cpd_order(φ)-1`.

If the RBF centers do not equal the the `features`, you can use the keyword argument `centers` to
pass a list of centers. If `φ` is a vector, then the length of `centers` and `φ` must be equal and 
`centers[i]` will be used in conjunction with `φ[i]` to build a `ShiftedKernel`. \
If `features` has 1D data, the output of the model will be a 1D-vector.
If it should be a scalar instead, set the keyword argument `vector_output` to `false`.
"""
function RBFModel(  
        features :: AnyVec{ FType },
        labels :: AnyVec{ LType },
        φ :: Union{RadialFunction,AnyVec{<:RadialFunction}},
        poly_deg :: Int = 1;
        centers :: AnyVec{ CType } = [],
        vector_output :: Bool = true,
        use_static_arrays :: Bool = true
    ) where {FType, LType, CType}

    ## Basic Data integrity checks
    @assert !isempty(features) "Provide at least 1 feature vector."
    @assert !isempty(labels) "Provide at least 1 label vector."
    num_vars = length(features[1])
    num_outputs = length(labels[1])
    @assert all( length(s) == num_vars for s ∈ features ) "All features must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ labels ) "All labels must have same dimension."
    
    if !isempty( centers )
        @assert all( length(s) == num_vars for s ∈ centers ) "All centers must have dimension $(num_vars)."
    else
        centers = features
    end

    ## prepare provided training data
    ## are we using static arrays?
    static = ( 
        ( FType <: Number || FType <: StaticArray ) && 
        ( LType <: Number || LType <: StaticArray ) && 
        ( CType <: Number || CType <: StaticArray )
    )

    sites = ensure_vec_of_vecs(features, Val(static))
    values = ensure_vec_of_vecs(labels, Val(static))
    centers = ensure_vec_of_vecs(centers, Val(static))
    
    kernels = make_kernels( φ, centers )
    poly_deg = min( poly_deg, cpd_order(φ) - 1 )
    poly_basis = canonical_basis( num_vars, poly_deg )

    w, λ = coefficients( sites, values, kernels, poly_basis )

    ## build output polynomials
    poly_vec = StaticPolynomials.Polynomial[] 
    for coeff_ℓ ∈ eachcol( λ )
        push!( poly_vec, StaticPolynomials.Polynomial( poly_basis'coeff_ℓ ) )
    end 
    poly_sys = PolynomialSystem( poly_vec )

    ## build RBF system 
    num_centers = length(sites)
    rbf_sys = RBFSum(kernels, w)
  
    ## vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : true
     
    return RBFModel{vec_output}( kernels, polys )
end




# The `RBFInterpolationModel` constructor takes data sites and values and return an `RBFModel` that 
# interpolates these points.
# We allow for passing scalar data and transform it internally.


"""
    RBFInterpolationModel( sites :: Vector{VS}, values :: Vector{VT}, φ, poly_deg = 1; 
        static_arrays = nothing, vector_output = true ) where {VS<:NumberOrVector, VT<:NumberOrVector}

Return an RBFModel `m` that is interpolating, i.e., `m(sites[i]) == values[i]` for all 
`i = eachindex(sites)`.
`φ` should be a `RadialFunction` or a vector of `RadialFunction`s that has the same length 
as `sites` and `values`.
`poly_deg` specifies the degree of the multivariate polynomial added to the RBF model.
It will be reset if needed.
`static_arrays` is automatically set to `true` if unspecified and the data dimensions are small.
`vector_output` is ignored if the `values` have length > 1. Elsewise it specifies whether to return 
vectors or scalars when evaluating.
"""
function RBFInterpolationModel(  
    s̃ides :: Vector{ VecTypeS },
    ṽalues :: Vector{ VecTypeV },
    φ :: Union{RadialFunction,Vector{<:RadialFunction}},
    poly_deg :: Int = 1;
    static_arrays :: Union{Bool, Nothing} = nothing,
    vector_output :: Bool = true,
    ) where { VecTypeS<:NumberOrVector, VecTypeV<:NumberOrVector }

    ## data integrity checks
    @assert length(s̃ides) == length(ṽalues) "Provide as many data sites as data labels."
    @assert !isempty(s̃ides) "Provide at least 1 data site."
    num_vars = length(s̃ides[1])
    num_outputs = length(ṽalues[1])
    @assert all( length(s) == num_vars for s ∈ s̃ides ) "All sites must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ ṽalues ) "All values must have same dimension."
    
    ## use static arrays? if no user preference is set …
    if isnothing(static_arrays)
        ## … use only if matrices are small
        static_arrays = (num_vars <= 10 && num_outputs <= 10)
    end

    ## prepare provided training data
    ## use same precision everywhere ( at least half-precision )
    TypeS = eltype( VecTypeS )
    TypeV = eltype( VecTypeV )
    dtype = promote_type( TypeS, TypeV, Float16 )
    NewVecTypeS = static_arrays ? SVector{ num_vars, dtype } : Vector{dtype}
    NewVecTypeV = static_arrays ? SVector{ num_outputs, dtype } : Vector{dtype}
    sites = convert_list_of_vecs( NewVecTypeS, s̃ides )
    values = convert_list_of_vecs( NewVecTypeV, ṽalues )
    
    kernels = make_kernels( φ, sites )
    poly_deg = min( poly_deg, cpd_order(φ) - 1 )
    poly_basis = canonical_basis( num_vars, poly_deg )

    w, λ = coefficients( sites, values, kernels, poly_basis )

    ## build output polynomials
    poly_vec = StaticPolynomials.Polynomial[] 
    for coeff_ℓ ∈ eachcol( λ )
        push!( poly_vec, StaticPolynomials.Polynomial( poly_basis'coeff_ℓ ) )
    end 
    poly_sys = PolySystem{static_arrays}( poly_vec, num_outputs )

    ## vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : true
    
    ## build RBF system 
    num_centers = length(sites)
    rbf_sys = RBFOutputSystem{static_arrays}(kernels, w, num_outputs, num_centers)
   
    return RBFModel{static_arrays, vec_output}( 
        rbf_sys, poly_sys, num_vars, num_outputs, num_centers
    )
end

# We want to provide an alternative constructor for interpolation models 
# so that the radial function can be defined by passing a `Symbol` or `String`.

const SymbolToRadialConstructor = NamedTuple((
    :gaussian => Gaussian,
    :multiquadric => Multiquadric,
    :inv_multiquadric => InverseMultiquadric,
    :cubic => Cubic,
    :thin_plate_spline => ThinPlateSpline
))

function RBFInterpolationModel(
        s̃ides :: Vector{ <: NumberOrVector }, 
        ṽalues :: Vector{ <:NumberOrVector },
        radial_func :: Union{Symbol, String}, 
        constructor_args :: Union{Nothing, Vector{<:Tuple}, Tuple} = nothing, 
        poly_deg :: Int = 1; kwargs ...
    )

    ## which radial function to use?
    radial_symb = Symbol( lowercase( string( radial_func ) ) )
    if !(radial_symb ∈ keys(SymbolToRadialConstructor))
        @warn "Radial Funtion $(radial_symb) not known, using Gaussian."
        radial_symb = :gaussian
    end
    constructor = SymbolToRadialConstructor[radial_symb]

    if isnothing(constructor_args)
        φ = constructor()
    elseif constructor_args isa Tuple 
        φ = constructor( constructor_args... )
    elseif constructor_args isa Vector 
        @assert length(constructor_args) == length(s̃ides)
        φ = [ constructor( arg_tuple... ) for arg_tuple ∈ constructor_args ]
    end
    
    return RBFInterpolationModel( s̃ides, ṽalues, φ, poly_deg; kwargs... )

end