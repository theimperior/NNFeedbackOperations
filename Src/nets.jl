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
# dependencies
# pkg> add Flux, BSON, NNlib, MAT, PyPlot

using Flux, Statistics
using Flux: onecold
using Printf, BSON
using Dates
using NNlib
using FeedbackNets
include("./dataManager.jl")
using .dataManager: make_batch
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
# TODO make use of unevaluated expressions and rewrite the name generation 
const config = "5digits" # 10debris 30debris, 50debris, 3digits, 4 digits, 5digits

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

function binarycrossentropy(y_hat, y)
	# splitting the computation of the binary crossentropy into two parts 
	# writing it in one equation would crash the script...
	a = -y .* log.(y_hat .+ eps(Float32))
	b = -(1 .- y) .* log.(1 .- y_hat .+ eps(Float32))
	c = a .+ b
	return sum(c) * 1 // length(y)
end

function onematch(y::AbstractVector, targets::AbstractVector)
	if ( length(y) != length(targets) ) @warn("vectors in onematch(y::AbstractVector, targets::AbstractVector) differ in length, results may be unexpected!") end
	return targets[Base.argmax(y)]
end

function onekill(y::AbstractVector)
	y[Base.argmax(y)] = 0 
	return y
end

function onematch!(y::AbstractMatrix, targets::AbstractMatrix) 
	matches = dropdims(mapslices(x -> onematch(x[1:(length(x) ÷ 2)], x[(length(x) ÷ 2 + 1):length(x)]), vcat(y, targets), dims=1), dims=1)
	y[:, :] = mapslices(x -> onekill(x), y, dims=1)
	return matches
end
onematch!(y::TrackedMatrix, targets::AbstractMatrix) = onematch!(Tracker.data(y), targets)

function trainReccurentNet(reccurent_model, train_set, test_set, model_cfg::String)
    function accuracy(data_set)
		acc = 0
		if( config == "10debris" || config == "30debris" || config == "50debris" )
			for (data, labels) in data_set
				# read the model output 1 times less, discard the output and read out again when calculating the onecold vector
				for i in 1:time_steps-1
					y_hat = reccurent_model(data)
				end
				acc += mean(onecold(reccurent_model(data)) .== onecold(labels))
				Flux.reset!(reccurent_model)
			end
			return acc / length(data_set)
		elseif ( config == "3digits" )
			for (data, labels) in data_set
				for i in 1:time_steps-1
					y_hat = reccurent_model(data)
				end
				y_hat = reccurent_model(data)
				matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
				acc += mean(matches .== 3)
				Flux.reset!(reccurent_model)
			end
			return acc / length(data_set)
		elseif ( config == "4digits" )
			for (data, labels) in data_set
				for i in 1:time_steps-1
					y_hat = reccurent_model(data)
				end
				y_hat = reccurent_model(data)
				matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
				acc += mean(matches .== 4)
				Flux.reset!(reccurent_model)
			end
			return acc / length(data_set)
		elseif ( config == "5digits" )
			for (data, labels) in data_set
				for i in 1:time_steps-1
					y_hat = reccurent_model(data)
				end
				y_hat = reccurent_model(data)
				matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
				acc += mean(matches .== 5)
				Flux.reset!(reccurent_model)
			end
			return acc / length(data_set)
		else
			
        end
    end
	
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
    for i in 1:epochs
		@printf("[%s] Epoch %d: Accuracy: %f, Loss: %f\n", Dates.format(now(), "HH:MM:SS"), i, accuracy(test_set), loss(test_set[1][1], test_set[1][2])) 
        Flux.train!(loss, params(reccurent_model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        if (rem(i, 20) == 0) 
			@printf("[%s] Epoch %d: Accuracy: %f, Loss: %f\n", Dates.format(now(), "HH:MM:SS"), i, accuracy(test_set), loss(test_set[1][1], test_set[1][2])) 
			# store intermediate model 
			# TODO check if intermediate model is available and load it! 
			if (i != epochs) BSON.@save "$(model_cfg)_$(config).$(i).bson" reccurent_model end
		end
    end
    return accuracy(test_set)
end


function trainFeedforwardNet(feedforward_model, train_set, test_set, model_cfg::String)
    function accuracy(data_set)
		acc = 0
		if( config == "10debris" || config == "30debris" || config == "50debris" )
			for (data, labels) in data_set
				acc += mean(onecold(feedforward_model(data)) .== onecold(labels))
			end
			return acc / length(data_set)
		elseif ( config == "3digits" )
			for (data, labels) in data_set
				y_hat = feedforward_model(data)
				matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
				acc += mean(matches .== 3)
			end
			return acc / length(data_set)
		elseif ( config == "4digits" )
			for (data, labels) in data_set
				y_hat = feedforward_model(data)
				matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
				acc += mean(matches .== 4)
			end
			return acc / length(data_set)
		elseif ( config == "5digits" )
			for (data, labels) in data_set
				y_hat = feedforward_model(data)
				matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
				acc += mean(matches .== 5)
			end
			return acc / length(data_set)
		else
		end
    end
	
	function loss(x, y)
		y_hat = feedforward_model(x)
		return binarycrossentropy(y_hat, y) + lambda * sum(norm, params(feedforward_model))
	end
    
    opt = Momentum(learning_rate, momentum)
    for i in 1:epochs
		@printf("[%s] Epoch %d: Accuracy: %f, Loss: %f\n", Dates.format(now(), "HH:MM:SS"), i, accuracy(test_set), loss(test_set[1][1], test_set[1][2])) 
        Flux.train!(loss, params(feedforward_model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        if (rem(i, 20) == 0) 
			@printf("[%s] Epoch %d: Accuracy: %f, Loss: %f\n", Dates.format(now(), "HH:MM:SS"), i, accuracy(test_set), loss(test_set[1][1], test_set[1][2])) 
			# store intermediate model 
			if (i != epochs) BSON.@save "$(model_cfg)_$(config).$(i).bson" feedforward_model end
		end
    end
    return accuracy(test_set)
end

@printf("Constructing models...\n")
BModel = spoerer_model_b(Float32, inputsize=(32, 32))
BKModel = spoerer_model_bk(Float32, inputsize=(32, 32))
BFModel = spoerer_model_bf(Float32, inputsize=(32, 32))
BLChain = spoerer_model_bl(Float32, inputsize=(32, 32), kernel=(3, 3), features=32)
BTChain = spoerer_model_bt(Float32, inputsize=(32, 32), kernel=(3, 3), features=32)
BLTChain = spoerer_model_bt(Float32, inputsize=(32, 32), kernel=(3, 3), features=32)

if usegpu
    BModel = gpu(BModel)
	BKModel = gpu(BKModel)
	BFModel = gpu(BFModel)
	BLChain = gpu(BLChain)
	BTChain = gpu(BTChain)
	BLTChain = gpu(BLTChain)
    hidden = Dict(key => gpu(val) for (key, val) in pairs(hidden))
end


BLModel = Flux.Recur(BLChain, hidden)
BTModel = Flux.Recur(BTChain, hidden)
BLTModel = Flux.Recur(BLTChain, hidden)

if(config == "10debris")
    train_folderpath = train_folderpath_debris
    train_filenames = ["5000_$(k)debris$(m).mat" for k=10, m in 1:20]
    test_folderpath = test_folderpath_debris
    test_filenames = ["5000_$(k)debris$(m).mat" for k=10, m in 1:2]
elseif(config == "30debris")
    train_folderpath = train_folderpath_debris
    train_filenames = ["5000_$(k)debris$(m).mat" for k=30, m in 1:20]
    test_folderpath = test_folderpath_debris
    test_filenames = ["5000_$(k)debris$(m).mat" for k=30, m in 1:2]
elseif(config == "50debris")
    train_folderpath = train_folderpath_debris
    train_filenames = ["5000_$(k)debris$(m).mat" for k=50, m in 1:20]
    test_folderpath = test_folderpath_debris
    test_filenames = ["5000_$(k)debris$(m).mat" for k=50, m in 1:2]
elseif(config == "3digits")
    train_folderpath = train_folderpath_digits
    train_filenames = ["5000_$(k)digits$(m).mat" for k=3, m in 1:20]
    test_folderpath = test_folderpath_digits
    test_filenames = ["5000_$(k)digits$(m).mat" for k=3, m in 1:2]
elseif(config == "4digits")
    train_folderpath = train_folderpath_digits
    train_filenames = ["5000_$(k)digits$(m).mat" for k=4, m in 1:20]
    test_folderpath = test_folderpath_digits
    test_filenames = ["5000_$(k)digits$(m).mat" for k=4, m in 1:2]
elseif(config == "5digits")
    train_folderpath = train_folderpath_digits
    train_filenames = ["5000_$(k)digits$(m).mat" for k=5, m in 1:20]
    test_folderpath = test_folderpath_digits
    test_filenames = ["5000_$(k)digits$(m).mat" for k=5, m in 1:2]
else
    @warn("Ups, somehting in the config went wrong...")
end



train_set, mean_img, std_img = make_batch(train_folderpath, train_filenames..., batch_size=batch_size)
# test_set needs to have the same batchsize as the train_set due to model state init
test_set, tmp1, tmp2 = make_batch(test_folderpath, test_filenames..., batch_size=batch_size)

if usegpu
    train_set = gpu.(train_set)
	test_set = gpu.(test_set)
end


@printf("loaded %d batches of size %d for training\n", length(train_set), size(train_set[1][1], 4))
@printf("loaded %d batches of size %d for testing\n", length(test_set), size(test_set[1][1], 4))

@info("Training BModel with $config\n")
best_acc = trainFeedforwardNet(BModel, train_set, test_set, "BModel")
BSON.@save "BModel_$config.bson" BModel best_acc

@info("Training BKModel with $config\n")
best_acc = trainFeedforwardNet(BKModel, train_set, test_set, "BKModel")
BSON.@save "BKModel_$config.bson" BKModel best_acc

@info("Training BFModel with $config\n")
best_acc = trainFeedforwardNet(BFModel, train_set, test_set, "BFModel")
BSON.@save "BFModel_$config.bson" BFModel best_acc

@info("Training BLModel with $config\n")
best_acc = trainReccurentNet(BLModel, train_set, test_set, "BLModel")
BSON.@save "BLModel_$config.bson" BLModel best_acc

@info("Training BTModel with $config\n")
best_acc = trainReccurentNet(BTModel, train_set, test_set, "BTModel")
BSON.@save "BTModel_$config.bson" BTModel best_acc

@info("Training BLTModel with $config\n")
best_acc = trainReccurentNet(BLTModel, train_set, test_set, "BLTModel")
BSON.@save "BLTModel_$config.bson" BLTModel best_acc

