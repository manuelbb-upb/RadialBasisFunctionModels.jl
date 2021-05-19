## TODO: Whenever I pass a type (e.g. `Int`), is it better to pass `R`?

"Drop-In Alternative to `StaticPolynomials.PolynomialSystem` when there are no outputs."
struct EmptyPolySystem{Nvars} end
Base.length(::EmptyPolySystem) = 0
StaticPolynomials.npolynomials(::EmptyPolySystem) = 0

"Evaluate for usual vector input. (Scalar input also supported, there are no checks)"
StaticPolynomials.evaluate(:: EmptyPolySystem, :: Union{R, Vector{R}}) where R<:Real = Int[]
"Evaluate for sized input."
StaticPolynomials.evaluate(:: EmptyPolySystem{Nvars}, :: StaticVector ) where {Nvars} = SVector{0,Int}()
(p :: EmptyPolySystem)( x :: NumberOrVector) = evaluate(p, x)

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
function StaticPolynomials.evaluate(
        :: ZeroPolySystem{Nvars, Nout}, :: StaticVector{S,R}) where {Nvars,Nout, S, R<:Real}
    return @SVector(zeros(R, Nout))
end
(p :: ZeroPolySystem)(x :: NumberOrVector) = evaluate(p, x)

function StaticPolynomials.jacobian( :: ZeroPolySystem{Nvars,Nout}, x:: Union{R, Vector{R}}) where{Nvars,Nout,R<:Real}
    zeros(Int, Nout, Nvars)
end
function StaticPolynomials.jacobian( :: ZeroPolySystem{Nvars,Nout}, x:: StaticVector) where{Nvars,Nout}
    @SMatrix(zeros(Int, Nout, Nvars))
end

function StaticPolynomials.evaluate_and_jacobian( p :: ZeroPolySystem, args...) 
    return  evaluate(p, args...), jacobian(p, args...) 
end

Base.getindex( :: EmptyPolySystem{Nvars}, :: Int ) where Nvars = ZeroPoly{Nvars}()

function StaticPolynomials.gradient( :: ZeroPoly{Nvars}, x:: Union{R, Vector{R}}) where{Nvars,R<:Real}
    zeros(Int, Nvars)
end

function StaticPolynomials.gradient( :: ZeroPoly{Nvars}, x :: StaticVector) where{Nvars}
    @SVector(zeros(Int, Nvars))
end

function StaticPolynomials.evaluate_and_gradient( z :: ZeroPoly, args... )
    return 0, gradient(z, args... )
end
