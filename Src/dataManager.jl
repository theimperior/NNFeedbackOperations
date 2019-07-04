module dataManager

using MAT 
using PyPlot
using Base.Iterators: repeated, partition
using Statistics
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

"""
    make_batch(filepath, batch_size=128, normalize=true)
    
Creates batches with size batch_size(default 128) from file at given filepath. Images will be normalized if normalize is set (default true). 
Structure of the .mat file: 

    fieldname | size
    ----------------
       images | N x width x height
  bin_targets | N x 10

where N denotes the number of samples

TODO: 
batch_size = -1 in this case create only one batch out of the data with lenght of the data
create wrapper for concatenating multiple .mat files into training batches
"""
# function make_batch(filepath; batch_size=128, normalize=true)
function make_batch(filepath, filenames...; batch_size=128, normalize=true)
    images = nothing
    bin_targets = nothing
    for (i, filename) in enumerate(filenames)
        # load the data from the mat file
        file = "$filepath$filename"
        @printf("Reading %d of %d from %s\n", i, length(filenames), file)
        matfile = matopen(file)
        # size(images) = (N, width, height, 1)
        imagepart = read(matfile, "images")
        # size(bin_targets) = (N, 10)
        bin_targetpart = read(matfile, "binary_targets")
        close(matfile) 
        if(isnothing(images))
            images = imagepart
            bin_targets = bin_targetpart
        else
            images = cat(dims=1, images, imagepart)
            bin_targets = cat(dims=1, bin_targets, bin_targetpart)   
        end
        
    end

    # rearrange the images array so it matches the convention of Flux width x height x channels x batchsize(Setsize)    
    images = permutedims(images, (2, 3, 4, 1))
    # rearrange binary targets: targetarray(10) x batchsize(Setsize)
    bin_targets = permutedims(bin_targets, (2, 1))
   
    @printf("Dimension of images (%d, %d, %d, %d)\n", size(images)...)
    @printf("Dimension of binary targets (%d, %d)\n", size(bin_targets)...)
    
    setsize = size(images, 4)
    images = convert(Array{Float64}, images) 
    
    mean_img = mean(images, dims=4)
    std_img = std(images, mean=mean_img, dims=4)
    if(normalize)
        @printf("normalize dataset...\n")
        std_img_tmp = std_img
        std_img_tmp[std_img_tmp.==0] .= 1
        for i in 1:setsize
            images[:, :, :, i] = (images[:, :, :, i] - mean_img) ./ std_img_tmp
        end
    end
    
    # Convert to Float32
    bin_targets = convert(Array{Float32}, bin_targets)
    images = convert(Array{Float32}, images) 
    mean_img = convert(Array{Float32}, mean_img)
    std_img = convert(Array{Float32}, std_img)
    # uncomment to display one sample of the images
    # matshow(dropdims(images[:,:,:,10], dims=3), cmap=PyPlot.cm.gray, vmin=0, vmax=255)

    @printf("Creating batches\n")
    # TODO Progressbar
    idxsets = partition(1:size(images, 4), batch_size)
    train_set = [make_minibatch(images, bin_targets, i) for i in idxsets];
    
    return train_set, mean_img, std_img
end # function make_batch
end # module dataManager