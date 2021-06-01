## TODO: Whenever I pass a type (e.g. `Int`), is it better to pass `R`? #src

# If the polynomial degree is < 0, we use an `EmptyPolySystem`:

"Drop-In Alternative to `StaticPolynomials.PolynomialSystem` when there are no outputs."
struct EmptyPolySystem{Nvars} end
Base.length(::EmptyPolySystem) = 0
StaticPolynomials.npolynomials(::EmptyPolySystem) = 0

"Evaluate for usual vector input. (Scalar input also supported, there are no checks)"
StaticPolynomials.evaluate(:: EmptyPolySystem, :: Union{R, Vector{R}}) where R<:Real = Int[]
"Evaluate for sized input."
StaticPolynomials.evaluate(:: EmptyPolySystem{Nvars}, :: StaticVector ) where {Nvars} = SVector{0,Int}()
(p :: EmptyPolySystem)( x :: NumberOrVector) = evaluate(p, x)

function StaticPolynomials.jacobian( :: EmptyPolySystem{Nvars}, args... ) where Nvars
    Matrix{Int}(undef, 0, Nvars )
end

function StaticPolynomials.evaluate_and_jacobian( p :: EmptyPolySystem, args ... )
    return p(args...), jacobian(p, args...)
end