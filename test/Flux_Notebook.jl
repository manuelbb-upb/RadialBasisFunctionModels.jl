### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 774b5c68-c2e8-11eb-0c25-c124276640d6
begin
	using Revise
	using Pkg
	Pkg.activate(tempname())
	Pkg.develop(url="https://github.com/manuelbb-upb/RBFModels.jl.git")
	Pkg.add("Flux")
	Pkg.add("MLDatasets")
	using Flux, MLDatasets
	using Random, Statistics
	using RBFModels
end

# ╔═╡ 8c51b60d-e1b5-439d-ab67-ef7813490684
using Flux.Optimise: update!

# ╔═╡ ec8e09e0-2238-4314-baf4-1209794bb4e4
md"Create some test data: `N_d` is the total number of samples."

# ╔═╡ a974f4de-47b2-4981-9381-717abd8ccda2
begin
	F = x -> [
		sum(x.^2);
		exp(x[1]) + sum(x);
		sin( sum(x)/(1+sum(x)) )
	]
		
	N_d = 100
	features = [ rand(3) for i = 1 : N_d ]
	labels = F.(features)
	data_indices = eachindex(features)
	
	# training data
	training_percentage = .8
	num_training = ceil(Int, N_d * training_percentage )
	num_test = N_d - num_training
	
	train_indices = randperm(N_d)[1:num_training]
	X_train = features[ train_indices ]
	Y_train = labels[ train_indices ]
	
	test_indices = setdiff( data_indices, train_indices )
	X_test = features[ test_indices ]
	Y_test = labels[ test_indices ]
	
	n_rbf_centers = ceil(Int, num_training/5)
	center_indices = randperm(num_training)[ 1 : n_rbf_centers ]
	nothing
end

# ╔═╡ 4b0db5f4-77cb-4391-a3bb-aae7d3c1c140
md"Build a model by chaining a dense neural net and an RBF Layer."

# ╔═╡ 1f80e0a3-71a3-483d-acc5-93c032655d4e
begin
	function get_rbf_model( m̃ )
		global X_train, Y_train
		X̃ = m̃.( X_train )
		rbf_centers = X̃[ center_indices ]
		return RBFModel( X̃, Y_train, Multiquadric(), -1; centers = rbf_centers )
	end
	
	function get_model1()
		n_out = 8
		layer1 = Dense(3, 10)
		layer2 = Dense(10, n_out)
	
		m̃ = Chain( layer1, layer2 )

		r = get_rbf_model( m̃ )

		return Chain(m̃, r)
	end
end

# ╔═╡ da1f41c7-12dd-4c8f-b267-a74bf99bc73b
begin
	batch_size = 10
	train_loader = Flux.Data.DataLoader((X_train, Y_train); batchsize = batch_size)
end

# ╔═╡ 44b9ab7e-c52e-4faa-b925-12eaf7c734da
Markdown.parse("""
## Minibatch Training

The training data is split into batches of size ``N_b = $(batch_size)``.
Suppose the labels have dimension ``k``.
The loss is calculated over the batch ``Y``:
```math
	L(Y) = \\frac{1}{N_b} \\sum_{y_i ∈ Y} \\frac{1}{k} \\sum_{ℓ=1}^k  (y_{i,ℓ} - \\hat{y}_{i,ℓ})^2
```
""")

# ╔═╡ 5cdfe543-d773-41ec-8840-2c79b767fb27
md"Training without affecting the rbf layer at all."

# ╔═╡ 10dc99e7-028b-4378-a549-c32b49f14a65
Flux.trainable(::RBFModel) = ()

# ╔═╡ 2ad4a6ab-57ff-45d6-9087-73bae6e81aff
begin
	opt = ADAM()
end

# ╔═╡ c8f1ce7d-a17f-4d13-acb5-e03d6002a273
begin
	m = get_model1()
	loss_sample = (x, y) -> Flux.Losses.mse(m(x), y);
	loss_batch = function( X, Y )
		mean( loss_sample.(X,Y) )
	end
	
	ps = params(m)	# not affecting RBF Layer 
	
	test_loss_before = loss_batch( X_test, Y_test )
	
	Flux.@epochs 2 for (X_batch, Y_batch) ∈ train_loader 
		loss_val, pback = Flux.Zygote.pullback(ps) do
			loss_batch(X_batch, Y_batch)
		end
		@show loss_val
		gs = pback( one(loss_val) )
		update!(opt, ps, gs )
	end
	test_loss_after = loss_batch( X_test, Y_test )
	
	Markdown.parse("$(test_loss_before) -> \n$(test_loss_after)")
end

# ╔═╡ 80b2d7ed-1cb6-46b1-a3da-17e19b434569
begin
	m̃ = get_model1()[1]
	
	loss_whole_chain = function(X,Y)
		local m
		rbf = get_rbf_model( m̃ )
		m = Chain( m̃, rbf )
		loss_sample = (x, y) -> Flux.Losses.mse(m(x), y);
		return mean( loss_sample.(X,Y) )
	end
	
	ps_2 = params(m̃)	# not affecting RBF Layer 
	
	test_loss_before_2 = loss_whole_chain( X_test, Y_test )
	
	Flux.@epochs 1 for (X_batch, Y_batch) ∈ train_loader 
		loss_val, pback = Flux.Zygote.pullback(ps_2) do
			loss_whole_chain(X_batch, Y_batch)
		end
		@show loss_val
		gs = pback( one(loss_val) )
		update!(opt, ps_2, gs )
	end
	test_loss_after_2 = loss_whole_chain( X_test, Y_test )
	
	Markdown.parse("$(test_loss_before_2) -> \n$(test_loss_after_2)")
end

# ╔═╡ 0123fe9d-352e-453c-8482-08c9230da47f
test_l = function(x)
	M = RBFModel( X_train, Y_train; centers = X_train[center_indices] )
	M(x, 1)
end

# ╔═╡ ed05ca08-9da5-48ad-8795-5ae6f3ee07dd
Flux.Zygote.gradient( test_l, rand(3))

# ╔═╡ 9f8ae0a4-adfb-49a9-8591-b3ba8a2f3409
M = RBFModel( X_train, Y_train; centers = X_train[center_indices] )

# ╔═╡ 589e6360-5d11-494b-9081-5ea30111b052
auto_grad(M, rand(3))

# ╔═╡ Cell order:
# ╠═774b5c68-c2e8-11eb-0c25-c124276640d6
# ╠═8c51b60d-e1b5-439d-ab67-ef7813490684
# ╟─ec8e09e0-2238-4314-baf4-1209794bb4e4
# ╠═a974f4de-47b2-4981-9381-717abd8ccda2
# ╟─4b0db5f4-77cb-4391-a3bb-aae7d3c1c140
# ╠═1f80e0a3-71a3-483d-acc5-93c032655d4e
# ╟─44b9ab7e-c52e-4faa-b925-12eaf7c734da
# ╠═da1f41c7-12dd-4c8f-b267-a74bf99bc73b
# ╟─5cdfe543-d773-41ec-8840-2c79b767fb27
# ╠═10dc99e7-028b-4378-a549-c32b49f14a65
# ╠═2ad4a6ab-57ff-45d6-9087-73bae6e81aff
# ╠═c8f1ce7d-a17f-4d13-acb5-e03d6002a273
# ╠═80b2d7ed-1cb6-46b1-a3da-17e19b434569
# ╠═0123fe9d-352e-453c-8482-08c9230da47f
# ╠═ed05ca08-9da5-48ad-8795-5ae6f3ee07dd
# ╠═9f8ae0a4-adfb-49a9-8591-b3ba8a2f3409
# ╠═589e6360-5d11-494b-9081-5ea30111b052
