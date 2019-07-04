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
using Flux: crossentropy, onecold
using Printf, BSON
import LinearAlgebra: norm
using NNlib
using FeedbackNets

include("./dataManager.jl")
using .dataManager: make_batch

using Base
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
usegpu = true
config = "10debris" # 30debris, 50debris, 3digits, 4 digits, 5digits
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

function trainReccurentNet(reccurent_model, train_set, test_set)
    function loss(x, y)
        loss_val = 0.0f0
        for i in 1:time_steps
            loss_val += crossentropy(reccurent_model(x), y) + lambda * sum(norm, params(reccurent_model))
        end
        Flux.reset!(reccurent_model)
        return loss_val
    end
    
    function accuracy(test_set)
        for i in 1:time_steps-1
            y_hat = reccurent_model(test_set[1])
        end
        acc = mean(onecold(reccurent_model(test_set[1])) .== onecold(test_set[2]))
        Flux.reset!(reccurent_model)
        return acc
    end
    
    opt = Momentum(learning_rate, momentum)
    for i in 1:epochs
        Flux.train!(loss, params(reccurent_model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        acc = accuracy(test_set)
        @printf("Accuracy %f in epoch %d\n", acc, i)
        flush(Base.stdout)
    end
    acc = accuracy(test_set)
    @printf("final accuracy: %d\n", accuracy(test_set))
    return acc
end

function trainFeedforwardNet(feedforward_model, train_set, test_set)
    function accuracy(test_set)
        return mean(onecold(feedforward_model(test_set[1])) .== onecold(test_set[2]))
    end
	
	function loss(x, y)
		loss_val = crossentropy(feedforward_model(x), y) + lambda * sum(norm, params(feedforward_model))
	    @printf("Loss: %f\n", loss_val)
	    return loss_val
	end
    
    opt = Momentum(learning_rate, momentum)
    for i in 1:epochs
        Flux.train!(loss, params(feedforward_model), train_set, opt)
        opt.eta = adapt_learnrate(i)
        acc = accuracy(test_set)
        @printf("Accuracy %f in epoch %d\n", acc, i)
        flush(Base.stdout)
    end
    acc = accuracy(test_set)
    @printf("Final accuracy on test set: %d\n", acc)
    return acc
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
    hidden = Dict(key => val |> gpu for (key, val) in pairs(hidden))
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
test_set, tmp1, tmp2 = make_batch(train_folderpath, test_filenames..., batch_size=batch_size)

if usegpu
    train_set = gpu.(train_set)
	test_set = gpu.(test_set)
end


@printf("loaded %d batches of size %d for training\n", length(train_set), size(train_set[1][1], 4))
@printf("loaded %d batches of size %d for testing\n", length(test_set), size(test_set[1][1], 4))

@info("Training BModel with $config\n")
best_acc = trainFeedforwardNet(BModel, train_set, test_set[1])
BSON.@save "BModel_$config.bson" BModel best_acc

@info("Training BKModel with $config\n")
best_acc = trainFeedforwardNet(BKModel, train_set, test_set[1])
BSON.@save "BKModel_$config.bson" BKModel best_acc

@info("Training BFModel with $config\n")
best_acc = trainFeedforwardNet(BFModel, train_set, test_set[1])
BSON.@save "BFModel_$config.bson" BFModel best_acc

@info("Training BLModel with $config\n")
best_acc = trainReccurentNet(BLModel, train_set, test_set[1])
BSON.@save "BLModel_$config.bson" BLModel best_acc

@info("Training BTModel with $config\n")
best_acc = trainReccurentNet(BTModel, train_set, test_set[1])
BSON.@save "BTModel_$config.bson" BTModel best_acc

@info("Training BLTModel with $config\n")
best_acc = trainReccurentNet(BLTModel, train_set, test_set[1])
BSON.@save "BLTModel_$config.bson" BLTModel best_acc

