using Pkg;
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using RadialBasisFunctionModels
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

DocMeta.setdocmeta!(RadialBasisFunctionModels, :DocTestSetup, :(using RadialBasisFunctionModels); recursive=true)

makedocs(;
    modules=[RadialBasisFunctionModels],
    authors="Manuel Berkemeier",
    repo="https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl/blob/{commit}{path}#{line}",
    sitename="RadialBasisFunctionModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl",
        assets=String[],
        mathengine = Documenter.MathJax2()
    ),
    pages=[
        "Home" => "index.md",
        "Main Module" => "RadialBasisFunctionModels.md"
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/RadialBasisFunctionModels.jl",
)

Pkg.activate(current_env)