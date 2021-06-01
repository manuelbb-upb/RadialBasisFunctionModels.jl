using Pkg;
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using RBFModels
using Documenter

include("make_literate.jl")

project_path = joinpath(@__DIR__, "..", "Project.toml")

test_path = joinpath(@__DIR__, "..", "test", "Project.toml")

if !(project_path ∈ Base.load_path())
    push!(LOAD_PATH, project_path)
end
if !(test_path ∈ Base.load_path())
    push!(LOAD_PATH, test_path)
end
#%%

DocMeta.setdocmeta!(RBFModels, :DocTestSetup, :(using RBFModels); recursive=true)

makedocs(;
    modules=[RBFModels],
    authors="Manuel Berkemeier",
    repo="https://github.com/manuelbb-upb/RBFModels.jl/blob/{commit}{path}#{line}",
    sitename="RBFModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://manuelbb-upb.github.io/RBFModels.jl",
        assets=String[],
        mathengine = Documenter.MathJax2()
    ),
    pages=[
        "Home" => "index.md",
        "Main Module" => "RBFModels.md"
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/RBFModels.jl",
)

Pkg.activate(current_env)