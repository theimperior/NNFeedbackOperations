module accuracy

using Flux, Statistics
using Flux: onecold


function binarycrossentropy(y_hat, y)
	# splitting the computation of the binary crossentropy into two parts 
	# writing it in one equation would crash the script...
	a = -y .* log.(y_hat .+ eps(Float32))
	b = -(1 .- y) .* log.(1 .- y_hat .+ eps(Float32))
	c = a .+ b
	return sum(c) * 1 // length(y)
end

function onekill(y::AbstractVector)
	y[Base.argmax(y)] = 0 
	return y
end

function onematch(y::AbstractVector, targets::AbstractVector)
	if ( length(y) != length(targets) ) @warn("vectors in onematch(y::AbstractVector, targets::AbstractVector) differ in length, results may be unexpected!") end
	return targets[Base.argmax(y)]
end

function onematch!(y::AbstractMatrix, targets::AbstractMatrix) 
	matches = dropdims(mapslices(x -> onematch(x[1:(length(x) ÷ 2)], x[(length(x) ÷ 2 + 1):length(x)]), vcat(y, targets), dims=1), dims=1)
	y[:, :] = mapslices(x -> onekill(x), y, dims=1)
	return matches
end

onematch!(y::TrackedMatrix, targets::AbstractMatrix) = onematch!(Tracker.data(y), targets)

function recur_accuracy(reccurent_model, data_set, time_steps, dataset_name::String)
	acc = 0
	for (data, labels) in data_set
		# read the model output 1 times less, discard the output and read out again when calculating the onecold vector
		for i in 1:time_steps-1
			y_hat = reccurent_model(data)
		end
		y_hat = reccurent_model(data)
		Flux.reset!(reccurent_model)
		
		if( dataset_name == "10debris" || dataset_name == "30debris" || dataset_name == "50debris" || dataset_name == "MNIST" ) 
			acc += mean(onecold(y_hat) .== onecold(labels))
		elseif ( dataset_name == "3digits" )
			# This has been tested and is executed one after the other
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 3)
		elseif ( dataset_name == "4digits" )
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 4)
		elseif ( dataset_name == "5digits" )
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 5)
		end
	end
	return round(acc / length(data_set), digits=5)
end

function ff_accuracy(feedforward_model, data_set, dataset_name::String)
	acc = 0
	for (data, labels) in data_set
		y_hat = feedforward_model(data)
		if( dataset_name == "10debris" || dataset_name == "30debris" || dataset_name == "50debris" || dataset_name == "MNIST" )
			acc += mean(onecold(y_hat) .== onecold(labels))
		elseif ( dataset_name == "3digits" )
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 3)
		elseif ( dataset_name == "4digits" )
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 4)
		elseif ( dataset_name == "5digits" )
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 5)
		end
	end
	return round(acc / length(data_set), digits=5)
end

end # module accuracy
