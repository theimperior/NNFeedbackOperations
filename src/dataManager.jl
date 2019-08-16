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
   
    @debug("Dimension of images $(size(images, 1)) x $(size(images, 2)) x $(size(images, 3)) x $(size(images, 4))")
    @debug("Dimension of binary targets $(size(bin_targets)) x $(size(bin_targets))")
    
    
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
    @debug("Creating batches")
    idxsets = partition(1:size(images, 4), batch_size)
    train_set = [make_minibatch(images, bin_targets, i) for i in idxsets];
    
    return train_set, mean_img, std_img
end # function make_batch


function load_MNIST(;batch_size=100, normalize_imgs=true, truncate_imgs=true)
	@debug("loading MNIST dataset")
		
	# process __train__ images
	mnist_trainlabels = MNIST.labels()
	mnist_trainimgs = MNIST.images()
	# reshape array to 28 x 28 x batchsize
	train_imgs = zeros(28, 28, 1, size(mnist_trainimgs, 1))
	for i in 1:size(mnist_trainimgs, 1)
		train_imgs[:,:,:,i] = mnist_trainimgs[i]
	end

	if(normalize_imgs)
	   mean_img, std_img = normalizePixelwise!(train_imgs, truncate=truncate_imgs)
   end
	
	bin_train_targets = convert(Array{Float32}, onehotbatch(mnist_trainlabels, 0:9))
	train_imgs = convert(Array{Float32}, train_imgs) 
	
	@debug("Creating train batches")
	idxsets = partition(1:size(train_imgs, 4), batch_size)
	train_set = [make_minibatch(train_imgs, bin_train_targets, i) for i in idxsets]
	
	# process __test__ images
	mnist_testlabels = MNIST.labels(:test)
	mnist_testimgs = MNIST.images(:test)
	# reshape array to 28 x 28 x batchsize
	test_imgs = zeros(28, 28, 1, size(mnist_testimgs, 1))
	for i in 1:size(mnist_testimgs, 1)
		test_imgs[:,:,:,i] = mnist_testimgs[i]
	end
	
	mean_img, std_img = normalizePixelwise!(test_imgs)
	
	bin_test_targets = convert(Array{Float32}, onehotbatch(mnist_testlabels, 0:9))
	test_imgs = convert(Array{Float32}, test_imgs) 
	
	@debug("Creating test batches")
	idxsets = partition(1:size(test_imgs, 4), batch_size)
	test_set = [make_minibatch(test_imgs, bin_test_targets, i) for i in idxsets]
	
	return train_set, test_set
end

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
