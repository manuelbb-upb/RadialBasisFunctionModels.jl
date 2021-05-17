## DP = DynamicPolynomials
## TODO: Whenever I pass a type (e.g. `Int`), is it better to pass `R`?

"Drop-In Alternative to `StaticPolynomials.PolynomialSystem` when there are no outputs."
struct EmptyPolySystem{Nvars} end
Base.length(::EmptyPolySystem) = 0

"Evaluate for usual vector input. (Scalar input also supported, there are no checks)"
StaticPolynomials.evaluate(:: EmptyPolySystem, :: Union{R, Vector{R}}) where R<:Real = Int[]
"Evaluate for sized input."
StaticPolynomials.evaluate(:: EmptyPolySystem{Nvars}, :: StatVec{R} ) where {Nvars,R<:Real} = SVector{0,Int}()
(p :: EmptyPolySystem)( x :: NumberOrVector) = evaluate(p, x)

"Constructor for `EmptyPolySystem` with `n` variables; intended for use when there is an empty polynomial array."
EmptyPolySystem( :: Vector{<:DP.Polynomial}, n :: Int ) = EmptyPolySystem{n}()

"Alternative to `PolynomialSystem` of an `RBFModel`."
struct ZeroPolySystem{Nvars,Nout} 
    polys :: EmptyPolySystem{Nvars}
    function ZeroPolySystem{Nvars,Nout}() where {Nvars,Nout}
        new{Nvars,Nout}( EmptyPolySystem{Nvars}() )
    end
end
struct ZeroPoly{Nvars} end
(::ZeroPoly)(args...) = 0
StaticPolynomials.evaluate(::ZeroPoly, args...) = 0

"Evaluate for usual vector input. (Scalar input also supported, there are no checks)"
StaticPolynomials.evaluate(:: ZeroPolySystem{Nvars, Nout}, :: Union{R, Vector{R}}) where {Nvars,Nout,R<:Real} = zeros(R, Nout)
"Evaluate for sized input."
StaticPolynomials.evaluate(:: ZeroPolySystem{Nvars, Nout}, :: StatVec{R}) where {Nvars,Nout,R<:Real} = @SVector(zeros(R, Nout))
(p :: ZeroPolySystem)(x :: NumberOrVector) = evaluate(p, x)

function StaticPolynomials.jacobian( :: ZeroPolySystem{Nvars,Nout}, x:: Union{R, Vector{R}}) where{Nvars,Nout,R<:Real}
    zeros(Int, Nout, Nvars)
end
function StaticPolynomials.jacobian( :: ZeroPolySystem{Nvars,Nout}, x:: StatVec{R}) where{Nvars,Nout,R<:Real}
    @SMatrix(zeros(Int, Nout, Nvars))
end

function StaticPolynomials.evaluate_and_jacobian( p :: ZeroPolySystem, args...) 
    return  evaluate(p, args...), jacobian(p, args...) 
end

Base.getindex( :: EmptyPolySystem{Nvars}, :: Int ) where Nvars = ZeroPoly{Nvars}()

function StaticPolynomials.gradient( :: ZeroPoly{Nvars}, x:: Union{R, Vector{R}}) where{Nvars,R<:Real}
    zeros(Int, Nvars)
end

function StaticPolynomials.gradient( :: ZeroPoly{Nvars}, x :: StatVec{R}) where{Nvars,R<:Real}
    @SVector(zeros(Int, Nvars))
end

function StaticPolynomials.evaluate_and_gradient( z :: ZeroPoly, args... )
    return 0, gradient(z, args... )
end
