module dataManager

using MAT 
using PyPlot
using Base.Iterators: repeated, partition
using Statistics
using ProgressMeter
using Printf

# export make_batch

# loads and processes training data into batches 
# Data order in the .mat file 
# images: N_samples x 32 x 32 x 1
# targets N_samples x 1
# bin_targets: N_samples x 10

# SOURCE: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
# Bundle images together with labels into batches 
# X should be of size Width x Height x 1 x Samples
# Y should be of size 10 x Samples (encoded as binary targets)
# X_batch is 32 x 32 x 1 x batchsize
# Y_batch is 10 x batchsize
function make_minibatch(X, Y, idxset)
    X_batch = Array{Float32}(undef, size(X, 1), size(X, 2), 1, length(idxset))
    Y_batch = Array{Float32}(undef, 10, length(idxset))
    for i in 1:length(idxset)
        
        X_batch[:, :, :, i] = Float32.(X[:, :, :, idxset[i]])
        Y_batch[:, i] = Float32.(Y[:, idxset[i]])
    end    
    # train_set is a set of tuples (x_batch, y_batch) - first bracket -> which tupel, second bracket -Y what of the tupel
    # train_set[1][1]

    return (X_batch, Y_batch)
end # function make_minibatch

function make_batch(;batch_size=128, filepath="../digitclutter/src/light_debris/light_debris_with_debris.mat", normalize=true)
    # load the data from the mat file
    @printf("Reading .mat file form source %s\n", filepath)
    file = matopen("../digitclutter/src/light_debris/light_debris_with_debris.mat")
    images = read(file, "images")
    targets = read(file, "targets")
    bin_targets = read(file, "binary_targets")
    close(file) 
    
    # rearrange the images array so it matches the convention of Flux width x height x channels x batchsize(Setsize)    
    images = permutedims(images, (2, 3, 4, 1))
    # rearrange binary targets: targetarray(10) x batchsize(Setsize)
    bin_targets = permutedims(bin_targets, (2, 1))
    
    setsize = size(images, 4)
    images = convert(Array{Float32}, images)
    bin_targets = convert(Array{Float32}, bin_targets)
    
    @printf("calculate mean and standart deviation of dataset\n")
    # calculate normalization matrix (mean, standard deviation)
    mean_img = mean(images, dims=4)
    std_img = std(images, mean=mean_img, dims=4)
    std_img_tmp = std_img
    std_img_tmp[std_img_tmp.==0] .= 1
    
    if(normalize) 
        p = Progress(setsize, 1) # what's 100% and minimum update interval: 1s
        for i in 1:setsize
            images[:, :, :, i] = (images[:, :, :, i] - mean_img) ./ std_img_tmp
            next!(p)
        end
    end
    
    # uncomment to display one sample of the images
    # matshow(dropdims(images[:,:,:,10], dims=3), cmap=PyPlot.cm.gray, vmin=0, vmax=255)

    @printf("Creating batches\n")
    # TODO Progressbar
    idxsets = partition(1:size(images, 4), batch_size)
    train_set = [make_minibatch(images, bin_targets, i) for i in idxsets];
    
    return train_set, mean_img, std_img
end # function make_batch
end # module dataManager