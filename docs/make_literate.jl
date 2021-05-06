
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(joinpath(@__DIR__))
using Literate 

Literate.markdown(
    joinpath( @__DIR__, "..", "src", "RBFModels.jl"), 
    joinpath( @__DIR__, "src" );    
    documenter = true,
    codefence = "````@example RBFModels" => "````"
    )

#%% make readme
Literate.markdown(
    joinpath( @__DIR__, "..", "test", "README.jl"), 
    joinpath( @__DIR__, ".." );    
    documenter = false,
    codefence = "````@example README" => "````"
)
#%%
Pkg.activate(current_env)