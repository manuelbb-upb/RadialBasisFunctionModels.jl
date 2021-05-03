using RBFModels
using Documenter

using Literate

Literate.markdown("../src/RBFModels.jl", "docs/src"; documenter = true)

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
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/RBFModels.jl",
)
