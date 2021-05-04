
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(joinpath(@__DIR__))
using Literate 

Literate.markdown(
    joinpath( @__DIR__, "..", "src", "RBFModels.jl"), 
    joinpath( @__DIR__, "src" ); documenter = true)
Pkg.activate(current_env)