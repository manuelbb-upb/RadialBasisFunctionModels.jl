using RBFModels

# Setup a 2D1D function to generate training values:
F = x -> sum( x.^2 )

# Generate data sites (a vector of vectors) …
sites = [ rand(Float32, 2) for i = 1 : 5 ]

# … and the values (a vector of numbers)
vals = F.(sites)

# Use the Multiquadric:
φ = Multiquadric()

#%%

rbf = RBFInterpolationModel( sites, vals, φ, 0 ; static_arrays = false)
