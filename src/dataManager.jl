module dataManager

using MAT 
using Base.Iterators: repeated, partition
using Statistics
using Flux.Data.MNIST
using Flux:onehotbatch
"""
	make_minibatch(X, Y, idxset)
	
loads and bundles training data and labels into batches 
X should be of size Width x Height x 1 x batchsize
Y should be of size 10 x batchsize (encoded as binary targets)
X_batch is 32 x 32 x 1 x batchsize
Y_batch is 10 x batchsize

SOURCE: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl
"""
function make_minibatch(X, Y, idxset)
    X_batch = Array{Float32}(undef, size(X, 1), size(X, 2), 1, length(idxset))
    Y_batch = Array{Float32}(undef, 10, length(idxset))
    for i in 1:length(idxset)
        
        X_batch[:, :, :, i] = Float32.(X[:, :, :, idxset[i]])
        Y_batch[:, i] = Float32.(Y[:, idxset[i]])
    end    
    return (X_batch, Y_batch)
end

"""
    make_batch(filepath, batch_size=128, normalize=true)
    
Creates batches with size batch_size(default 100) from filenames at given filepath. Images will be normalized if normalize is set (default true). 
If batch_size equals -1 the batch size will be the size of the dataset
Structure of the .mat file: 

    fieldname | size
    ----------------
       images | N x width x height
  bin_targets | N x 10

where N denotes the number of samples
"""
function make_batch(filepath, filenames...; batch_size=100, normalize_imgs=true, truncate_imgs=true)
    images = Array{Float64}(undef, 0)
    bin_targets = Array{Float64}(undef, 0)
    for (i, filename) in enumerate(filenames)
        # load the data from the mat file
        file = "$filepath$filename"
        @debug("Reading $(i) of $(length(filenames)) from $(file)")
        matfile = matopen(file)
        # size(images) = (N, width, height, 1)
        imagepart = read(matfile, "images")
        # size(bin_targets) = (N, 10)
        bin_targetpart = read(matfile, "binary_targets")
        close(matfile) 

        images = cat(dims=1, images, imagepart)
        bin_targets = cat(dims=1, bin_targets, bin_targetpart)   
        
    end

    # rearrange the images array so it matches the convention of Flux width x height x channels x batchsize(Setsize)    
    images = permutedims(images, (2, 3, 4, 1))
    # rearrange binary targets: targetarray(10) x batchsize(Setsize)
    bin_targets = permutedims(bin_targets, (2, 1))
   
    @debug("Dimension of images $(size(images))")
    @debug("Dimension of targets $(size(bin_targets))")
    
    
    images = convert(Array{Float64}, images) 
    
    if(normalize_imgs)
		(mean_img, std_img) = normalizePixelwise!(images, truncate_imgs)
	end
    
    # Convert to Float32
    bin_targets = convert(Array{Float32}, bin_targets)
    images = convert(Array{Float32}, images) 
    mean_img = convert(Array{Float32}, mean_img)
    std_img = convert(Array{Float32}, std_img)
	
    # display one sample of the images depends on PyPlot!
    # matshow(dropdims(images[:,:,:,10], dims=3), cmap=PyPlot.cm.gray, vmin=0, vmax=255)
	
	if ( batch_size == -1 ) 
	   batch_size = size(images, 4)
	end
	
    idxsets = partition(1:size(images, 4), batch_size)
    train_set = [make_minibatch(images, bin_targets, i) for i in idxsets];
    
    return train_set, mean_img, std_img
end # function make_batch


function make_MNIST_batch(;batch_size=100, normalize_imgs=true, truncate_imgs=true, create_validation_set=false)
	@debug("loading MNIST dataset")
	# process __train__ images
	mnist_trainlabels = MNIST.labels()
	mnist_trainimgs = MNIST.images()
	if(create_validation_set)
		train_set, t1, t2 = make_MNIST_minibatch(mnist_trainimgs[1:end-10000], mnist_trainlabels[1:end-10000], normalize_imgs)
		validation_set, t1, t2 = make_MNIST_minibatch(mnist_trainimgs[end-10000+1:end], mnist_trainlabels[end-10000+1:end], normalize_imgs)
	else 
		train_set, t1, t2 = make_MNIST_minibatch(mnist_trainimgs, mnist_trainlabels, normalize_imgs)
		validation_set = []
	end
	
	# process __test__ images
	mnist_testlabels = MNIST.labels(:test)
	mnist_testimgs = MNIST.images(:test)
	test_set, t1, t2 = make_MNIST_minibatch(mnist_testimgs, mnist_testlabels, normalize_imgs)
	
	return train_set, validation_set, test_set
end # function make_MNIST_batch 

function make_MNIST_minibatch(mnist_imgs, mnist_labels, normalize_imgs)
	# reshape array to 28 x 28 x batchsize
	imgs = zeros(28, 28, 1, size(mnist_imgs, 1))
	for i in 1:size(mnist_imgs, 1)
		imgs[:,:,:,i] = mnist_imgs[i]
	end
	# size(imgs) = (28, 28, 1, 60000) for the train set without validation! 
	
	if(normalize_imgs)
		mean_img, std_img = normalizePixelwise!(imgs)
	end
	
	bin_mnist_labels = convert(Array{Float32}, onehotbatch(mnist_labels, 0:9))
	imgs = convert(Array{Float32}, imgs) 
	
	@debug("Dimension of images $(size(imgs))")
    @debug("Dimension of labels $(size(mnist_labels))")
	
	idxsets = partition(1:size(imgs, 4), batch_size)
	data_set = [make_minibatch(imgs, bin_mnist_labels, i) for i in idxsets]
	
	return data_set, mean_img, std_img
end # function make_MNIST_minibatch

"""
normalize input images along the batch dimension
input should have standart flux order: Widht x height x channels x batchsize
if truncate is set to true the last 1% beyond 2.576 sigma will be clipped to 2.576 sigma
"""
function normalizePixelwise!(images; truncate=true)
	mean_img = mean(images, dims=4)
    std_img = std(images, mean=mean_img, dims=4)
	
	setsize = size(images, 4)
    
	@debug("normalize dataset")
	std_img_tmp = copy(std_img)
	std_img_tmp[std_img_tmp .== 0] .= 1
	for i in 1:setsize
		images[:, :, :, i] = (images[:, :, :, i] - mean_img) ./ std_img_tmp
	end
	if(truncate)
		# truncate the last 1% beyond 2.576 sigma 
		images[images .> 2.576] .= 2.576
		images[images .< -2.576] .= -2.576
	end
	return (mean_img, std_img)
end

end # module dataManager
