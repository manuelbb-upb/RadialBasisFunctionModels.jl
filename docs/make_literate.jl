
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(joinpath(@__DIR__))
using Literate

# TODO 
# re-insert autodocs block when https://github.com/JuliaDocs/Documenter.jl/issues/1192
# is fixed

#%% Replace include calls 
# Taken from Liteare.jl docs 
function replace_includes(str)

    included = ["radial_funcs.jl", "empty_poly_sys.jl", "constructors.jl", "derivatives.jl"]

    # Here the path loads the files from their proper directory,
    # which may not be the directory of the `examples.jl` file!
    path = joinpath( @__DIR__, "..", "src" )

    for ex in included
        content = read(joinpath(path,ex), String)
        str = replace(str, "include(\"$(ex)\")" => content)
    end
    return str
end

#%%

Literate.markdown(
    joinpath( @__DIR__, "..", "src", "RadialBasisFunctionModels.jl"), 
    joinpath( @__DIR__, "src" );    
    documenter = true,
    execute = false,
    codefence = "````@example RadialBasisFunctionModels" => "````",
    preprocess = replace_includes
    )

#%% make readme
#=Literate.markdown(
    joinpath( @__DIR__, "..", "test", "README.jl"), 
    joinpath( @__DIR__, "src" );    
    documenter = true,
    codefence = "````@example README" => "````"
)=#
Literate.markdown(
    joinpath( @__DIR__, "..", "test", "README_template.jl"), 
    joinpath( @__DIR__, ".." );    
    documenter = false,
    execute = false,
    #codefence = "````julia" => "````",
    name = "README"
)

Literate.script(
    joinpath( @__DIR__, "..", "test", "README_template.jl"), 
    joinpath( @__DIR__, "..", "test" );    
    name = "README"
)

#%%
Pkg.activate(current_env)