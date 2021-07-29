
# This file is included from within RadialBasisFunctionModels.jl #src 

# ## Getting the Coefficients
const VecOfVecs{T} = AbstractVector{<:AbstractVector}

# ###  Polynomial Basis 

# Any constructor of an `RBFModel` must solve for the coefficients in ``\eqref{eqn:coeff_basic}``.
# To build the equation system, we need a basis ``\{p_j\}_{1 \le j \le Q}`` of ``Π_d(ℝ^n)``.
# For the interpolation system to be solvable we have to choose the 
# right polynomial space for ``p``.
# Basically, if the RBF Kernel (or the radial function) is 
# *conditionally positive definite* of order ``D`` we have to 
# find a polynomial ``p`` with ``\deg p \ge D-1``.[^wendland]
# If the kernel is CPD of order ``D=0`` we do not have to add an polynomial 
# and can interpolate arbitrary (distinct) data points. \

# The canonical basis is ``x_1^{α_1} x_2^{α_2} … x_n^{α_n}`` with 
# ``α_i ≥ 0`` and ``Σ_i α_i ≤ d``.
# For ``\bar{d} \le d`` we can recursively get the non-negative integer solutions for 
# ``Σ_i α_i = \bar{d}`` with the following function:

@doc """
    non_negative_solutions( d :: Int, n :: Int)

Return a matrix with columns that correspond to solution vectors 
``[x_1, …, x_n]`` to the equation ``x_1 + … + x_n = d``,
where the variables are non-negative integers.
"""
function non_negative_solutions( d :: Int, n :: Int ) :: Matrix{Int}
    if n == 1
        return fill(d,1,1)
    else
        num_sols = binomial( d + n - 1, n - 1)
        sol_matrix = Matrix{Int}(undef, n, num_sols)
        j = 1
        for i = 0 : d
            ## find all solutions of length `n-1` that sum to `i` 
            ## if we add `d-i` to each column, then each column 
            ## has `n` elements and sums to `d`
            padded_shorter_solutions = vcat( d-i, non_negative_solutions(i, n-1) )
            num_shorter_sols = size( padded_shorter_solutions, 2 )
            sol_matrix[:, j : j + num_shorter_sols - 1] .= padded_shorter_solutions
            j += num_shorter_sols
        end
        return sol_matrix
    end
end

# The polyonmial basis exponents are then given by all possible 
# ``\bar{d}\le d``:
@doc """
    non_negative_solutions_ineq( d :: Int, n :: Int)

Return a matrix with columns that correspond to solution vectors 
``[x_1, …, x_n]`` to the equation ``x_1 + … + x_n <= d``,
where the variables are non-negative integers.
"""
@memoize ThreadSafeDict function non_negative_solutions_ineq( d :: Int, n :: Int ) :: Matrix{Int}
    no_cols = binomial( n + d, n)
    ret_mat = Matrix{Int}(undef, n, no_cols )

    i = 1
    for d̄ = 0 : d
        sub_mat = non_negative_solutions( d̄, n)
        ret_mat[:, i:i + size(sub_mat,2) - 1] .= sub_mat[:,:]
        i += size(sub_mat, 2)
    end
    
    return ret_mat
end

# !!! note 
#     I did an unnecessary rewrite of `non_negative_solutions` to be 
#     Zygote-compatible. Therefore the matrices etc.  
#     `Combinatorics` has `multiexponents` which should do the same...

# We **don't** use `DynamicPolynomials.jl` to generate the Polyomials **anymore**.
# Zygote did overflow when there were calculations with those polynomials.
# Not a problem for calculating the basis (because of we are `ignore()`ing 
# the basis calculation now, assuming we never want to differentiate 
# with respect to `n,d`),
# but when constructing the outputs from them.
# Instead we directly construct `StaticPolynomial`s and define a 
# `PolynomialSystem` that evaluates all basis polynomials.

@doc """
    canonical_basis( n:: Int, d :: Int ) :: Union{PolynomialSystem, EmptyPolySystem}

Return the canonical basis of the space of `n`-variate 
polynomials of degree at most `d`.
"""
function canonical_basis(n :: Int, d::Int, OneType :: Type = Float64)
    if d < 0
        return EmptyPolySystem{n}()
    else
        exponent_matrix = non_negative_solutions_ineq( d, n )
        one_float = OneType(1)  # `one_float` is used as coefficient(s) to guarantee floating point output
        return PolynomialSystem(
             ( Polynomial( [one_float,], e[:,:] ) for e ∈ eachcol(exponent_matrix) )... 
        )
    end
end

# ### Solving the Equation System 

# Now let ``\{p_j\}_{1\le j\le Q}`` be a basis of the polynomial space.
# Set ``P = [ p_j(x^i) ] ∈ ℝ^{N × Q}`` and ``Φ = φ(\| x^i - x^j \|)``.
# In case of interpolation, the linear equation system for the 
# coefficients of $r$ is 
# ```math
# S c := \begin{equation}
#     \begin{bmatrix}
#     Φ & P \\
#     P^T & 0_{Q × Q}
#     \end{bmatrix}
#     \begin{bmatrix}
#         w \\
#         λ
#     \end{bmatrix}
#     \stackrel{!}=
#     \begin{bmatrix}
#     Y
#     \\
#     0_Q
#     \end{bmatrix}.
#     \tag{I}
#     \label{eqn:coeff}
# \end{equation}
# ```

# We can also use differing feature vectors and centers.
# ``Φ`` then becomes 
# ``Φ = [k_j(x^i)]_{1\le i \le N_d, 1\le j \le N_c} = [φ(‖ x^i - ξ^j ‖)]``,
# where we denote the number of kernel centers by ``N_c`` and the number
# of feauters (``d``ata) by ``N_d``.
# In the overdetermined least squares case (with pair-wise different centers and 
# pair-wise different features), we do away with the second row of equations in \eqref{eqn:coeff}.
# The solution ``c = [w, λ]^T`` is then given by the
# Moore-Penrose Pseudo-Inverse:
# ```math 
#     c = \underbrace{ ( S^T S )^{-1} S^T}_{=S^\dagger} \begin{bmatrix}
#     Y
#     \\
#     0_Q
#     \end{bmatrix}.
# ```
# Julia automatically computes the LS solution with `S\RHS`.

# !!! note 
#     When we have vector data ``Y ⊂ ℝ^k``, e.g. from modelling MIMO functions, then 
#     Julia easily allows for multiple columns in the righthand side of the interpolation 
#     equation system and we get weight vectors for multiple models, that can 
#     be thought of as one vector model ``r\colon ℝ^n \to ℝ^k``.

# Some helpers to build the matrices:

"Provided an array of sites with `n` variables, return a polynomial basis system of degree `poly_deg`."
function make_polys(sites, poly_deg = 1)
    poly_precision = promote_type(Float16, inner_type(sites))
    poly_basis_sys = Zyg.ignore() do 
        canonical_basis( length(sites[1]), poly_deg, poly_precision )
    end
    return poly_basis_sys
end

function make_kernel( φ :: RadialFunction, center :: AbstractVector{<:Real})
    return ShiftedKernel(φ,center)
end

"Return array of `ShiftedKernel`s based functions in `φ_arr` with centers from `centers`."
function make_kernels( φ_arr :: AbstractVector{<:RadialFunction}, centers :: VecOfVecs )
    @assert length(φ_arr) == length(centers)
    return [ make_kernel(φ,c) for (φ,c) ∈ zip( φ_arr, centers) ]
end

"Return array of `ShiftedKernel`s based function `φ` with centers from `centers`."
function make_kernels( φ :: RadialFunction, centers :: VecOfVecs )
    return [ make_kernel(φ,c) for c ∈ centers ]
end

"Return RBF basis matrix by applying each kernel to each site. Kernels vary with the columns."
function _rbf_matrix( kernels, sites )
    return transpose( hcat( map(kernels, sites)... ) )
end

"Return polynomial basis matrix by applying each basis polynomial to each site. Polynomials vary with the columns."
function _poly_matrix( polys, sites )
    return transpose( hcat( map(polys, sites)... ) ) 
end

"Return RBF and Polynomial basis matrices as well as the vector of kernels and the polynomial basis system."
function get_matrices( φ, sites, centers = []; poly_deg = 1 )

    rbf_centers = isempty(centers) ? sites : centers

    kernels = make_kernels( φ, rbf_centers )
    polys = make_polys( sites, poly_deg )

    return _rbf_matrix( kernels, sites ), _poly_matrix( polys, sites ), kernels, polys
end 

# Later, the construtor will call the above helper functions and then use 
# these methods to retrieve the coefficients:

@doc """
    coefficients(sites, values, kernels, rad_funcs, polys )

Return the coefficient matrices `w` and `λ` for an rbf model 
``r(x) = Σ_{i=1}^N wᵢ φ(\\|x - x^i\\|) + Σ_{j=1}^M λᵢ pᵢ(x)``,
where ``N`` is the number of centers and ``M``
is the number of n-variate basis polynomials.
"""
function coefficients( Φ, P, values;  mode :: Symbol = :ls)
   
    N_d, N_c = size( Φ )
    N_dp, Q = size( P )
    
    @assert N_d == N_dp "Row count of RBF and Poly basis matrices does not match."

    if N_d < N_c 
        error("Underdetermined models not supported yet.")
    end
    if N_d < Q 
        error("Too few data sites for selectod polynomial degree. (Need at least $(Q).)")
    end
    
    ## system matrix S and right hand side
    S = [Φ P]
    RHS = transpose( hcat(values... ) )

    return _coefficients( Φ, P, S, RHS, Val(:ls) )
end

# The actual work is delegated to `_coefficients`.

function _coefficients( Φ, P, S, RHS, ::Val{:ls} )
    N_c = size(Φ,2); Q = size(P,2);
    coeff = S \ RHS 
    return _coeff_matrices(coeff, S, RHS, N_c, Q )
end

## interpolation requires zero padding
function _coefficients( Φ, P, S, RHS, ::Val{:interpolation} )
    N_d, N_c = size(Φ); Q = size(P,2);
    @assert N_d == N_c "Interpolation requires same number of features and centers." # TODO remove assertion
    S̃ = [ S ;                               # N_d × (N_c + Q)
          P' zeros(eltype(S), Q, Q )]       # Q × N_d and Q × Q 
    RHS_padded = [ RHS;
        zeros( eltype(RHS), Q, size(RHS,2) ) ];
    coeff = S̃ \ RHS_padded 
    return _coeff_matrices( coeff, S̃, RHS_padded, N_c, Q )
end

## treat sized arrays in a special way, so that zero padding preserves the types
## can potentially be removed once https://github.com/JuliaArrays/StaticArrays.jl/issues/856 is resolved
 function _coefficients( Φ, P, S :: StaticMatrix, RHS :: StaticMatrix, ::Val{:interpolation} )
    N_d, N_c = size(Φ); Q = size(P,2);
    @assert N_d == N_c "Interpolation requires same number of features and centers." # TODO remove assertion

    S̃ = [S ; 
         P' @SMatrix(zeros(eltype(S),Q,Q)) ];
    RHS_padded = [ RHS;
        @SMatrix(zeros(eltype(RHS), Q ,size(RHS,2)))];
    coeff = S̃ \ RHS_padded 
    return _coeff_matrices( coeff, S̃, RHS_padded, N_c, Q )
end

## this method returns random coefficients
function _coefficients( Φ, P, S, RHS, ::Val{:rand} )
    F = Base.promote_eltype(S, RHS)
    N_c = size(Φ,2); Q = size(P,2)
    m = N_c + Q
    k = size(RSH, 2)
    return _coeff_matrices(rand( F, m, k ), S, RHS, N_c, Q )
end

# We want statically sized matrices when appropriate, that is why we 
# call `_coeff_matrices`:
"Return rbf weights, polynomial regressor weights, system matrix and right hand side of the RBF equations."
function _coeff_matrices(coeff :: AbstractMatrix, S, RHS, N_c, Q )
    return view(coeff, 1 : N_c, :), view(coeff, N_c + 1 : N_c + Q, :), S, RHS
end 

## index into `coeff` using static array so that the results are sized, too
function _coeff_matrices(coeff :: StaticMatrix, S, RHS, N_c, Q )
    return coeff[ SVector{N_c}(1 : N_c), :], coeff[ SVector{Q}( N_c + 1 : N_c + Q ), :], S, RHS
end

# We can easily impose linear equality constraints,
# for example requiring interpolation only on a subset of features.
# In matrix form, $I$ linear equality constraints (for ``k`` outputs) can be written 
# as 
# ```math
# E c = b, \quad E ∈ ℝ^{I×(N_c + Q)}, b ∈ ℝ^{I\times k},\, I,k ∈ ℕ_0. 
# ```
# Now, let $ĉ$ be the least squares solution from above.
# The constrained solution is 
# ```math 
#  c = ĉ - Ψ E^T ( E Ψ E^T)^{-1} ( E ĉ - b ), \; Ψ := (S^T S)^{-1}
# \tag{CLS1}
# \label{eqn:cls1}
# ```
# This results from forming the Lagrangian of an equivalent minimization problem.
# Let ``δ = ĉ - c ∈ ℝ^{q\times k}, q = N_c + Q,`` and define the constraint residuals 
# as ``γ = Eĉ - b ∈ ℝ^{I\times k}``.
# The Lagrangian for minimizing ``δ^TS^TSδ`` under ``Eδ=γ`` is 
# ```math 
# \begin{aligned}
#     L &= δ^T S^T S δ + 2 λ^T( E δ - γ )\\
#     D_δL &= 2 δ^T S^T S + 2λ^T E \\
#     D_λL &= 2 δ^T E^T - 2 γ^T 
# \end{aligned}
# ```
# Setting the derivatives to zero leads to \eqref{eqn:cls1} via 
# ```math 
#     \begin{bmatrix}
#         S^T S & E^T \\
#         E & 0_{I\times I}
#     \end{bmatrix}
#     \begin{bmatrix}
#     δ \\ λ 
#     \end{bmatrix}
#     = \begin{bmatrix}
#     0_{q\times k} \\ γ
#     \end{bmatrix}
# \tag{L}
# \label{eqn:cls2}
# ```
# See [^adv_eco] for details.

# TODO make this respect StaticMatrices too!! #src
function constrained_coefficients( 
        w :: AbstractMatrix{<:Real}, 
        λ :: AbstractMatrix{<:Real}, 
        S :: AbstractMatrix{<:Real},
        E :: AbstractMatrix{<:Real},
        b :: AbstractMatrix{<:Real}
    )
    ## Using Lagrangian approach:
    
    ĉ = [w; λ]  # least squares solution
    γ = E*ĉ - b # constraint residuals

    I, q = size(E)
    k = size(w,2)

    A = vcat(
        [S'S E'],
        [E zeros(Int,I,I)]
    )

    RHS = [ 
        zeros(Int, q, k);
        γ 
    ]

    δλ = A \ RHS 
    δ = δλ[1 : q, :]

    c = ĉ - δ  # coefficients for constrained problem

    N_c = size(w,1)
 
    return c[1 : N_c, :], c[N_c+1:end, :]
end

# For the case that mentioned above, that is, interpolation at a 
# subset of sites, we can easily build the ``E`` matrix from the ``S`` 
# matrix by taking the corresponding rows.
function constrained_coefficients( 
        w :: AbstractMatrix{<:Real}, 
        λ :: AbstractMatrix{<:Real}, 
        S :: AbstractMatrix{<:Real},
        RHS_ls :: AbstractMatrix{<:Real},
        interpolation_indices :: AbstractVector{Int} 
    )

    E = S[interpolation_indices, :]
    b = RHS_ls[interpolation_indices, :]
    return constrained_coefficients( w, λ, S, E, b )
end
    
# ### The Actual, Usable Constructor 

# We want the user to be able to pass 1D data as scalars and use the following helpers: 

ensure_vec_of_vecs( before :: AbstractVector{<:AbstractVector{<:Real}} ) = before
ensure_vec_of_vecs( before :: AbstractVector{ <:Real }) = [[x,] for x in before ]

function inner_type( vec_of_vecs :: AbstractVector{<:AbstractVector{T}}) where T
    if Base.isabstracttype(T)   # like Any if data is of mixed precision
        return Float64 
    else
        return T
    end
end

function _check_data(features,labels,centers; type_checks = true)
    ## Basic Data integrity checks

    @assert !isempty(features) "Provide at least 1 feature vector."
    @assert !isempty(labels) "Provide at least 1 label vector."
    num_vars = length(features[1])
    num_outputs = length(labels[1])
    @assert all( length(s) == num_vars for s ∈ features ) "All features must have same dimension."
    @assert all( length(v) == num_outputs for v ∈ labels ) "All labels must have same dimension."
    
    num_sites = length(features)
    num_vals = length(labels)
    @assert num_sites == num_vals "Provide as many features as labels."

    if type_checks
        sites = ensure_vec_of_vecs(features)
        values = ensure_vec_of_vecs(labels)
        if !isempty(centers)
            @assert all( length(c) == num_vars for c ∈ centers ) "All centers must have dimension $(num_vars)."
            C = ensure_vec_of_vecs(centers)
        else
            C = sites
        end
    else
        sites = features
        values = labels
        C = centers 
    end

    num_centers = length(C)

    return num_vars, num_outputs, num_sites, num_vals, num_centers, sites, values, C 
end

# We now have all ingredients for the basic outer constructor:


@doc """
    RBFModel( features, labels, φ = Multiquadric(), poly_deg = 1; kwargs ... )

Construct a `RBFModel` from the feature vectors in `features` and 
the corresponding labels in `lables`, where `φ` is a `RadialFunction` or a vector of 
`RadialFunction`s.\n
Scalar data can be used, it is transformed internally. \n
StaticArrays can be used, e.g., `features :: Vector{<:SVector}`. 
Providing `SVector`s only might speed up the construction.\n
If the degree of the polynomial tail, `poly_deg`, is too small it will be set to `cpd_order(φ)-1`.

If the RBF centers do not equal the the `features`, you can use the keyword argument `centers` to
pass a list of centers. If `φ` is a vector, then the length of `centers` and `φ` must be equal and 
`centers[i]` will be used in conjunction with `φ[i]` to build a `ShiftedKernel`. \n
If `features` has 1D data, the output of the model will be a 1D-vector.
If it should be a scalar instead, set the keyword argument `vector_output` to `false`.
"""
function RBFModel( 
        features :: AbstractVector{ <:NumberOrVector },
        labels :: AbstractVector{ <:NumberOrVector },
        φ :: Union{RadialFunction, AbstractVector{<:RadialFunction}} = Multiquadric(),
        poly_deg :: Int = 1;
        centers :: AbstractVector{ <:NumberOrVector } = Vector{Float16}[],
        interpolation_indices :: AbstractVector{ <: Int } = Int[],
        vector_output :: Bool = true,
        coeff_mode :: Symbol = :auto,
        save_matrices :: Bool = true
    )

    num_vars, num_outputs, num_sites, num_vals, num_centers, sites, values, C = _check_data( features, labels, centers )
   
    if coeff_mode == :auto
        can_interpolate_uniquely = φ isa RadialFunction ? poly_deg >= cpd_order(φ) - 1 : all( poly_deg >= cpd_order(phi) - 1 for phi in φ )
        coeff_mode = num_sites == num_centers && can_interpolate_uniquely ? :interpolation : :ls
    end

    Φ, P, kernels, poly_basis_sys = get_matrices( φ, sites, C; poly_deg )

    w, λ, S, RHS = coefficients( Φ, P, values; mode = coeff_mode )

    meta = save_matrices ? (rbf_mat = Φ, poly_mat = P) : nothing

    if !isempty(interpolation_indices)
        w, λ = constrained_coefficients( w, λ, S, RHS, interpolation_indices)
    end
    
    ## build output polynomials
    poly_sum = PolySum( poly_basis_sys, transpose(λ) )

    ## build RBF system 
    rbf_sys = RBFSum(kernels, transpose(w), num_outputs)
  
    ## vector output? (dismiss user choice if labels are vectors)
    vec_output = num_outputs == 1 ? vector_output : true
     
    return RBFModel(rbf_sys, poly_sum, num_vars, num_outputs, num_centers, meta; vec_output)
end

# ### Special Constructors

# We offer some specialized models (that simply wrap the main type).
struct RBFInterpolationModel
    model :: RBFModel 
end
(mod :: RBFInterpolationModel)(args...) = mod.model(args...)
@forward RBFInterpolationModel.model grad, jac, jacT, auto_grad, auto_jac

# The constructor is a tiny bit simpler and additional checks take place:
"""
    RBFInterpolationModel(features, labels, φ, poly_deg; kwargs… )

Build a model interpolating the feature-label pairs.
Does not accept `center` keyword argument.
"""
function RBFInterpolationModel( 
        features :: AbstractVector{ <:NumberOrVector },
        labels :: AbstractVector{ <:NumberOrVector },
        φ :: Union{RadialFunction,AbstractVector{<:RadialFunction}} = Multiquadric(),
        poly_deg :: Int = 1;
        vector_output :: Bool = true,
        save_matrices :: Bool = true
    )
    @assert length(features) == length(labels) "Provide as many features as labels!"
    
    if poly_deg < cpd_order(φ) - 1
        @warn "Polyonmial degree too small for interpolation. Using $(cpd_order(φ)-1)." 
        poly_deg = max( poly_deg,  cpd_order(φ) - 1 )
    end

    mod = RBFModel(features, labels, φ, poly_deg; vector_output, save_matrices, coeff_mode = :interpolation)
    return RBFInterpolationModel( mod )
end

# We want to provide a convenient alternative constructor for interpolation models 
# so that the radial function can be defined by passing a `Symbol` or `String`.

const SymbolToRadialConstructor = NamedTuple((
    :gaussian => Gaussian,
    :multiquadric => Multiquadric,
    :inv_multiquadric => InverseMultiquadric,
    :cubic => Cubic,
    :thin_plate_spline => ThinPlateSpline
))

"Obtain a `RadialFunction` from its name and constructor arguments."
function _get_rad_func( φ_symb :: Union{Symbol, String}, φ_args )

    ## which radial function to use?
    radial_symb = Symbol( lowercase( string( φ_symb ) ) )
    if !(radial_symb ∈ keys(SymbolToRadialConstructor))
        @warn "Radial Funtion $(radial_symb) not known, using Gaussian."
        radial_symb = :gaussian
        φ_args = nothing
    end
    
    constructor = SymbolToRadialConstructor[radial_symb]
    if isnothing(φ_args)
        φ = constructor()
    else
        φ = constructor( φ_args... )
    end

    return φ
end

# The alternative constructors are build programmatically:
for op ∈ [ :RBFInterpolationModel, :RBFModel ]
    @eval begin
        function $op( 
                features :: AbstractVector{ <:NumberOrVector },
                labels :: AbstractVector{ <:NumberOrVector },
                φ_symb :: Union{Symbol, String},
                φ_args = nothing,
                poly_deg :: Int = 1; kwargs...
            )

            φ = _get_rad_func( φ_symb, φ_args )
            return $op(features, labels, φ, poly_deg; kwargs... )
        end
    end
end


# ### Container with Training Data

# The RBF Machine is similar in design to what an MLJ machine does:
# Training data (feature and label **vectors**) are stored and can be added.
# The inner model is trained with `fit!`.

# **TODO** In the future, we can customize the `fit!` method when updating a model 
# to only consider *new* training data. 
# This also makes type conversion of the whole data arrays unnecessary.

"""
    RBFMachineWithKernel(; φ, features, labels, poly_deg)

A container holding an inner `model :: RBFModel` (or `model == nothing`).
The inner model interpolates the data after calling `fit!`.
An array of arrays of features is stored in the `features` field.
Likewise for `labels`. 
After `add_data!`, the model has to be fitted again, indicated by its field `is_valid`.

In constrast to the more general RBFModel, only one single RBF function (i.e. one set of shape parameters)
should be used.
"""
@with_kw mutable struct RBFMachineWithKernel{
        FT <: AbstractVector{<:AbstractVector{<:AbstractFloat}},
        LT <: AbstractVector{<:AbstractVector{<:AbstractFloat}},
        R <: RadialFunction,
    }

    φ :: R = Gaussian()

    features :: FT = Vector{Float64}[]
    labels :: LT = Vector{Float64}[]
    
    poly_deg :: Int = 1

    model :: Union{Nothing, RBFModel} = nothing
    valid :: Bool = false   # is model trained on all data sites?

    function RBFMachineWithKernel{FT,LT,R}(φ::R, features::FT, labels::LT, poly_deg, model, valid ) where{R,FT,LT}
        @assert length(features) == length(labels) "Need same number of `features` and `labels`."
        features_copied = copy(features)
        labels_copied = copy(labels)
        return new{FT,LT,R}(φ,features_copied, labels_copied, poly_deg, model, valid )
    end

end

function RBFMachine(; kernel_name :: Symbol = :gaussian, kernel_args = nothing, poly_deg = 1, 
    labels :: FT = Vector{Float64}[], features :: LT = Vector{Float64}[], kwargs... ) where {FT, LT}
    T = promote_type(eltype(eltype(FT)), eltype(eltype(LT)))
    KARGS = isnothing(kernel_args) ? nothing : T.(kernel_args)
    φ = _get_rad_func( kernel_name, KARGS )
    @assert poly_deg >= cpd_order(φ) - 1 "Polynomial degree too low for interpolation."
    return RBFMachineWithKernel(; φ, poly_deg, labels, features, kwargs...) 
end   

"Return floating point type of training data elements."
_precision( :: RBFMachineWithKernel{FT,LT} ) where {FT,LT} = eltype( Base.promote_eltype(FT, LT) )

"Fit `mach :: RBFMachineWithKernel` to the training data."
function fit!( mach :: RBFMachineWithKernel )::Nothing
    @assert length(mach.features) > 0 "Cannot `fit!` without data."
    @assert length(mach.features) == length(mach.labels) "Interpolation requires same number of features and labels."
    num_needed =  binomial( mach.poly_deg + length(mach.features[1]), mach.poly_deg) 
    @assert length(mach.features) >= num_needed "Too few data sites for selected polynomial degree (need $(num_needed))."
    
    mach.model = RBFModel( mach.features, mach.labels, mach.φ, mach.poly_deg; save_matrices = true )

    mach.valid = true
    return nothing
end

# Forward evaluation methods of inner model:
( mach :: RBFMachineWithKernel )(args...) = mach.model(args...)
@forward RBFMachineWithKernel.model grad, jac, jacT, auto_grad, auto_jac

# Methods to add features and labels:
"Add a feature vector(s) and a label(s) to the `machine` container."
function add_data!( 
        m :: RBFMachineWithKernel, features :: AbstractVector{<:AbstractVector}, labels :: AbstractVector{<:AbstractVector}
    ) :: Nothing
    @assert length(features) == length(labels) "Provide same number of features and labels."
    @assert all( length(f) == length(features[1]) for f in features ) "Features must have same length."
    @assert all( length(l) == length(labels[1]) for l in labels ) "Labels must have same length"
    @assert isempty(m.features) || length(m.features[1]) == length(features[1]) && length(m.labels[1]) == length(labels[1]) "Length doesnt match previous data."
    append!(m.features, features)
    append!(m.labels, labels)
    m.valid = false
    return nothing
end

function add_data!(
        m :: RBFMachineWithKernel, feature :: AbstractVector{<:AbstractFloat}, label:: AbstractVector{<:AbstractFloat}
    ) :: Nothing 
    return add_data!(m, [ feature, ], [label, ])
end

# Convenience methods to "reset" a machine:

function Base.empty!( m :: RBFMachineWithKernel ) :: Nothing
    empty!(m.features)
    empty!(m.labels)
    m.model = nothing
    m.valid = false 
    return nothing
end

function Base.isempty(m :: RBFMachineWithKernel ) :: Bool 
    isempty( m.features ) && isempty( m.labels ) && isnothing(m.model)
end