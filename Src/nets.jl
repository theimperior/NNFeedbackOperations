"""
Author: Sebastian Vendt, University of Ulm

This script implements the four different neural networks proposed in the paper 
Recurrent Convolutional Neural Networks: A Better Model of Biological Object Recognition
of Courtney J. Spoerer et al. 

BNet:  the bottom up network with two hidden convolutional layers
BLNet: the BNet including lateral connections within the hidden layers
BTNet: the BNet including top down connections from the second hidden layer to the first
BLTNet:the BNet including top down and lateral connections 



"""
using Flux, Statistics
using Flux: onecold
using Printf, BSON
using Dates
using NNlib
using FeedbackNets
include("./dataManager.jl")
include("./accuracy.jl")
using .dataManager: make_batch
using .accuracy: binarycrossentropy, recur_accuracy, ff_accuracy
import LinearAlgebra: norm
norm(x::TrackedArray{T}) where T = sqrt(sum(abs2.(x)) + eps(T)) 


######################
# PARAMETERS
######################
const batch_size = 100
const momentum = 0.9f0
const lambda = 0.0005f0
init_learning_rate = 0.1f0
learning_rate = init_learning_rate
const epochs = 100
const decay_rate = 0.1f0
const decay_step = 40
# number of timesteps the network is unrolled
const time_steps = 4
const usegpu = true
const printout_interval = 5
# enter the datasets and models you want to train
const datasets = ["10debris", "30debris", "50debris", "3digits", "4digits", "5digits"]
const FFModels = [] # ["BModel", "BKModel", "BFModel"]
const FBModels = ["BLTModel"] # ["BTModel", "BLModel", "BLTModel"]

train_folderpath_debris = "../digitclutter/digitdebris/trainset/mat/"
train_folderpath_digits = "../digitclutter/digitclutter/trainset/mat/"
test_folderpath_debris = "../digitclutter/digitdebris/testset/mat/"
test_folderpath_digits = "../digitclutter/digitclutter/testset/mat/"
# end of parameters

if usegpu
    using CuArrays
end

hidden = Dict(
    "l1" => zeros(Float32, 32, 32, 32, 100),
    "l2" => zeros(Float32, 16, 16, 32, 100)
    )

function adapt_learnrate(epoch_idx)
    return init_learning_rate * decay_rate^(epoch_idx / decay_step)
end

function load_dataset(dataset_name)
	# generate filenames 
	train_filenames = ["5000_$(dataset_name)$(m).mat" for m in 1:20]
	test_filenames = ["5000_$(dataset_name)$(m).mat" for m in 1:2]
	if(dataset_name == "10debris" || dataset_name == "30debris" || dataset_name == "50debris")
		train_folderpath = train_folderpath_debris
		test_folderpath = test_folderpath_debris
	elseif(dataset_name == "3digits" || dataset_name == "4digits" || dataset_name == "5digits")
		train_folderpath = train_folderpath_digits
		test_folderpath = test_folderpath_digits
	end

	train_set, mean_img, std_img = make_batch(train_folderpath, train_filenames..., batch_size=batch_size)
	# test_set needs to have the same batchsize as the train_set due to model state init
	test_set, tmp1, tmp2 = make_batch(test_folderpath, test_filenames..., batch_size=batch_size)

	if usegpu
		train_set = gpu.(train_set)
		test_set = gpu.(test_set)
	end
	
	# TODO make use of debug logging for verbose output of the dataset loader
	# @printf("loaded %d batches of size %d for training\n", length(train_set), size(train_set[1][1], 4))
	# @printf("loaded %d batches of size %d for testing\n", length(test_set), size(test_set[1][1], 4))
	
	return (train_set, test_set)
end

function trainReccurentNet(reccurent_model, train_set, test_set, model_name::String)
	function loss(x, y)
        loss_val = 0.0f0
        for i in 1:time_steps
            loss_val += binarycrossentropy(reccurent_model(x), y)
        end
		Flux.reset!(reccurent_model)
		loss_val /= time_steps
		loss_val += lambda * sum(norm, params(reccurent_model))
        return loss_val
    end
    
    opt = Momentum(learning_rate, momentum)
	@printf("[%s] INIT with Accuracy: %f and Loss: %f\n", Dates.format(now(), "HH:MM:SS"), recur_accuracy(reccurent_model, test_set, config), loss(test_set[1][1], test_set[1][2])) 
    for i in 1:epochs
        Flux.train!(loss, params(reccurent_model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        if (rem(i, printout_interval) == 0)
			@printf("[%s] Epoch %d: Accuracy: %f, Loss: %f\n", Dates.format(now(), "HH:MM:SS"), i, recur_accuracy(reccurent_model, test_set, config), loss(test_set[1][1], test_set[1][2])) 
			# store intermediate model 
			# TODO check if intermediate model is available and load it! 
			if (i != epochs) BSON.@save "$(model_name)_$(config).$(i).bson" reccurent_model end
		end
    end
    return recur_accuracy(reccurent_model, test_set, config)
end


function trainFeedforwardNet(feedforward_model, train_set, test_set, model_name::String)
	function loss(x, y)
		y_hat = feedforward_model(x)
		return binarycrossentropy(y_hat, y) + lambda * sum(norm, params(feedforward_model))
	end
    
    opt = Momentum(learning_rate, momentum)
	@printf("[%s] INIT with Accuracy: %f and Loss: %f\n", Dates.format(now(), "HH:MM:SS"), ff_accuracy(feedforward_model, test_set, config), loss(test_set[1][1], test_set[1][2])) 
    for i in 1:epochs
        Flux.train!(loss, params(feedforward_model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        if (rem(i, printout_interval) == 0) 
			@printf("[%s] Epoch %d: Accuracy: %f, Loss: %f\n", Dates.format(now(), "HH:MM:SS"), i, ff_accuracy(feedforward_model, test_set, config), loss(test_set[1][1], test_set[1][2])) 
			# store intermediate model 
			if (i != epochs) BSON.@save "$(model_name)_$(config).$(i).bson" feedforward_model end
		end
    end
    return ff_accuracy(feedforward_model, test_set, config)
end

@printf("Constructing models...\n")
Models = [spoerer_model_b(Float32, inputsize=(32, 32)), 
			spoerer_model_bk(Float32, inputsize=(32, 32)),
			spoerer_model_bf(Float32, inputsize=(32, 32)),
			spoerer_model_bt(Float32, inputsize=(32, 32)),
			spoerer_model_bl(Float32, inputsize=(32, 32)),
			spoerer_model_blt(Float32, inputsize=(32, 32))]

if usegpu
	Models = gpu.(Models)
    hidden = Dict(key => gpu(val) for (key, val) in pairs(hidden))
end


Models[4] = Flux.Recur(Models[4], hidden)
Models[5]  = Flux.Recur(Models[5], hidden)
Models[6]  = Flux.Recur(Models[6], hidden)

for (idx, model_name) in enumerate(FFModels)
	for dataset_name in datasets
		@printf("Training $(model_name) with $(dataset_name)\n")
		(train_set, test_set) = load_dataset(dataset_name)
		best_acc = trainFeedforwardNet(Models[idx], train_set, test_set, model_name)
		BSON.@save "BModel_$config.bson" Models[idx] best_acc
	end
end

for (idx, model_name) in enumerate(FBModels)
	for dataset_name in datasets
		@printf("Training $(model_name) with $(dataset_name)\n")
		(train_set, test_set) = load_dataset(dataset_name)
		best_acc = trainReccurentNet(Models[idx+3], train_set, test_set, model_name)
		BSON.@save "BModel_$config.bson" Models[idx+3] best_acc
	end
end
