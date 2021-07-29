using RadialBasisFunctionModels
RBF = RadialBasisFunctionModels

sites = [rand(2) for i = 1 : 5]
vals = [rand(2) for i = 1 : 5]

#%%

r = RBFMachine(; features = sites, labels = vals)