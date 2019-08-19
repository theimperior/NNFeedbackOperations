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
using BSON
using Dates
using Printf
using NNlib
using FeedbackNets
include("./dataManager.jl")
include("./accuracy.jl")
using .dataManager: make_batch, make_minibatch, make_MNIST_batch
using .accuracy: binarycrossentropy, recur_accuracy, ff_accuracy
using Logging
import LinearAlgebra: norm
norm(x::TrackedArray{T}) where T = sqrt(sum(abs2.(x)) + eps(T)) 


######################
# PARAMETERS
######################
const batch_size = 100
const momentum = 0.9f0
const lambda = 0.0005f0
init_learning_rate = 0.1f0 
epochs = 100
const decay_rate = 0.1f0
const decay_step = 40
# number of timesteps the network is unrolled
const time_steps = 4
const usegpu = true
const printout_interval = 1
const time_format = "HH:MM:SS"
const date_format = "dd_mm_yyyy"
image_size = (32, 32) # MNIST is using 28, 28
# enter the datasets and models you want to train
dataset_names = ["10debris", "30debris", "50debris", "3digits", "4digits", "5digits", "MNIST"]
FFModel_names = ["BModel", "BKModel", "BFModel"]
FBModel_names = ["BTModel", "BLModel", "BLTModel"]

train_folderpath_debris = "../digitclutter/digitdebris/trainset/mat/"
train_folderpath_digits = "../digitclutter/digitclutter/trainset/mat/"
test_folderpath_debris = "../digitclutter/digitdebris/testset/mat/"
test_folderpath_digits = "../digitclutter/digitclutter/testset/mat/"

const model_save_location = "../trainedModels/"
const log_save_location = "../logs/"
# end of parameters

# DEBUGGING
# activate debugging with JULIA_DEBUG=modulename, with keyword all for all files or Main for this script only
# add DEBUG_ to log file name when debugging is active 
debug_str = ""
@debug begin
	global debug_str
	debug_str = "DEBUG_"
	"------DEBUGGING ACTIVATED------"
end

io = nothing

if usegpu
    using CuArrays
end

# Usage:
# ARG1 learning rate
# ARG2 epochs
# ARG3 dataset names
# ARG4 ff model names
# ARG5 fb model names
 
if (length(ARGS) > 0)
	init_learning_rate = parse(Float32, ARGS[1])
	epochs = parse(Int32, ARGS[2])
	dataset_names = convert.(String, split(ARGS[3]))
	FFModel_names = convert.(String, split(ARGS[4]))
	FBModel_names = convert.(String, split(ARGS[5]))
end

function adapt_learnrate(epoch_idx)
    return init_learning_rate * decay_rate^(epoch_idx / decay_step)
end

function load_dataset(dataset_name)
	global image_size
	
	if(dataset_name == "MNIST")
		image_size = (28,28)
		train_set, validation_set, test_set = make_MNIST_batch(create_validation_set=true)
	else
		image_size = (32, 32)
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
		validation_set = []
	end
	
	if usegpu
		train_set = gpu.(train_set)
		validation_set = gpu.(validation_set)
		test_set = gpu.(test_set)
	end

	@debug("loaded $(length(train_set)) batches of size $(size(train_set[1][1], 4)) for training")
	@debug("loaded $(length(validation_set)) batches of size $(size(validation_set[1][1], 4)) for validation")
	@debug("loaded $(length(test_set)) batches of size $(size(test_set[1][1], 4)) for testing")
	
	return (train_set, validation_set, test_set)
end

function trainReccurentNet(model, train_set, test_set, model_name::String, dataset_name::String)
	function loss(x, y)
        loss_val = 0.0f0
        for i in 1:time_steps
            loss_val += binarycrossentropy(model(x), y)
        end
		Flux.reset!(model)
		loss_val /= time_steps
		loss_val += lambda * sum(norm, params(model))
        return loss_val
    end
	
	function loss(dataset)
		loss_val = 0.0f0
		for (data, labels) in dataset
			loss_val += Tracker.data(loss(data, labels))
		end
		return loss_val / length(dataset)
	end
    
    opt = Momentum(init_learning_rate, momentum)
	@printf(io, "[%s] INIT with Accuracy(test_set): %.4f and Loss(test_set): %f\n", Dates.format(now(), time_format), recur_accuracy(model, test_set, time_steps, dataset_name), loss(test_set)) 
    for i in 1:epochs
        flush(io)
		Flux.train!(loss, params(model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        if ( rem(i, printout_interval) == 0 )
			@printf(io, "[%s] Epoch %3d: Accuracy: %.4f, Loss: %f\n", Dates.format(now(), time_format), i, recur_accuracy(model, train_set, time_steps, dataset_name), loss(train_set))
		end
    end
	@printf(io, "[%s] FINAL with Accuracy(test_set): %.4f and Loss(test_set): %f\n", Dates.format(now(), time_format), recur_accuracy(model, test_set, time_steps, dataset_name), loss(test_set)) 
    return recur_accuracy(model, test_set, time_steps, dataset_name)
end

function trainFeedforwardNet(model, train_set, test_set, model_name::String, dataset_name::String)
	function loss(x, y)
		y_hat = model(x)
		return binarycrossentropy(y_hat, y) + lambda * sum(norm, params(model))
	end
	
	function loss(dataset)
		loss_val = 0.0f0
		for (data, labels) in dataset
			loss_val += loss(data, labels)
		end
		return loss_val / length(dataset)
	end
    
    opt = Momentum(init_learning_rate, momentum)
	@printf(io, "[%s] INIT with Accuracy(test_set): %.4f and Loss(test_set): %f\n", Dates.format(now(), time_format), ff_accuracy(model, test_set, dataset_name), loss(test_set)) 
    for i in 1:epochs
		flush(io)
        Flux.train!(loss, params(model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        if ( rem(i, printout_interval) == 0 ) 
			@printf(io, "[%s] Epoch %3d: Accuracy: %.4f, Loss: %f\n", Dates.format(now(), time_format), i, ff_accuracy(model, train_set, dataset_name), loss(train_set)) 
		end
		# TODO store intermediate model or load if it already exists, one needs to find a good solution not to move the model on the cpu store it and then move it back to the gpu 
    end
	@printf(io, "[%s] FINAL with Accuracy(test_set): %.4f and Loss(test_set): %f\n", Dates.format(now(), time_format), ff_accuracy(model, test_set, dataset_name), loss(test_set)) 
	# return the accuracy across the test set
    return ff_accuracy(model, test_set, dataset_name)
end

FFModels = Dict( "BModel" => :spoerer_model_b, 
				 "BKModel" => :spoerer_model_bk,
				 "BFModel" => :spoerer_model_bf )
				 
FBModels = Dict( "BTModel" => :spoerer_model_bt,
				 "BLModel" => :spoerer_model_bl,
				 "BLTModel" => :spoerer_model_blt )



for model_name in FFModel_names
	# create a own log file for every model and all datasets
	global io
	fp = "$(log_save_location)$(debug_str)log_$(model_name).log"
	io = open(fp, "a+")
	global_logger(SimpleLogger(io)) # for debug outputs
	@printf(Base.stdout, "Logging to File: %s", fp)
	@printf(io, "\n--------[%s %s]--------\n", Dates.format(now(), date_format), Dates.format(now(), time_format))
	# dump configuration 
	@debug begin
		for symbol in names(Main)
			var = "$(symbol) = $(eval(symbol))"
			@printf(io, "%s\n", var)
		end
		"--------End of VAR DUMP--------"
	end
	flush(io)
	flush(Base.stdout)
	for dataset_name in dataset_names
		@printf(io, "[%s] Training %s with %s\n", Dates.format(now(), time_format), model_name, dataset_name)
		(train_set, validation_set, test_set) = load_dataset(dataset_name)
		
		# make sure the model gets recreated for every new dataset
		model = eval(get(FFModels, model_name, nothing))(Float32, inputsize=image_size)
		if (usegpu) model = gpu(model) end
		
		best_acc = trainFeedforwardNet(model, train_set, validation_set, model_name, dataset_name)
		model = cpu(model)
		BSON.@save "$(model_save_location)$(model_name)_$(dataset_name).bson" model best_acc
	end
	close(io)
end

for model_name in FBModel_names
	global io
	fp = "$(log_save_location)$(debug_str)log_$(model_name).log"
	io = open(fp, "a+")
	global_logger(SimpleLogger(io)) # for debug outputs
	@printf(Base.stdout, "Logging to File: %s", fp)
	@printf(io, "\n--------[%s %s]--------\n", Dates.format(now(), date_format), Dates.format(now(), time_format))
	# dump configuration 
	@debug begin
		for symbol in names(Main)
			var = "$(symbol) = $(eval(symbol))"
			@printf(io, "%s\n", var)
		end
		"--------End of VAR DUMP--------"
	end
	flush(io)
	flush(Base.stdout)
	for dataset_name in dataset_names
		@printf(io, "[%s] Training %s with %s\n", Dates.format(now(), time_format), model_name, dataset_name)
		(train_set, validation_set, test_set) = load_dataset(dataset_name)
		
		# make sure the model gets recreated for every new dataset
		hidden = Dict(
				"l1" => zeros(Float32, image_size[1], image_size[2], 32, batch_size),
				"l2" => zeros(Float32, image_size[1] ÷ 2, image_size[2] ÷ 2, 32, batch_size)
				)
		if usegpu
			hidden = Dict(key => gpu(val) for (key, val) in pairs(hidden))
		end
		
		chain = eval(get(FBModels, model_name, nothing))(Float32, inputsize=image_size)
		model = Flux.Recur(chain, hidden)
		if (usegpu) model = gpu(model) end
		
		best_acc = trainReccurentNet(model, train_set, validation_set, model_name, dataset_name)
		model = cpu(model)
		BSON.@save "$(model_save_location)$(model_name)_$(dataset_name).bson" model best_acc
	end
	close(io)
end

