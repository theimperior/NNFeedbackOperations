module accuracy

using Flux, Statistics
using Flux: onecold

export binarycrossentropy, onematch!, recur_accuracy, ff_accuracy

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
	matches = dropdims(mapslices(x -> onematch(x[1:(length(x) ? 2)], x[(length(x) ? 2 + 1):length(x)]), vcat(y, targets), dims=1), dims=1)
	y[:, :] = mapslices(x -> onekill(x), y, dims=1)
	return matches
end

"""
	onematch!(y::TrackedMatrix, targets::AbstractMatrix)
	
This function returns the value in targets at the position where y has its maximum value. 
The maximum itself is determined columnwise. Both input matrices should be of the same size! 
Once called the maximum value in y is set to zero! Calling this function multiple times will 
return the value in targets for the highest value in y, then for the second highest value and so on.
"""
onematch!(y::TrackedMatrix, targets::AbstractMatrix) = onematch!(Tracker.data(y), targets)

"""
	recur_accuracy(recurrent_model, data_set, time_steps, dataset_name::String)
	
Calculates the accuracy for a given recurrent model on the given dataset. The parameter time_steps defines the time at which the modeloutput is evaluated. 
The parameter dataset_name referes to the type of the dataset and can be one of the following:
10debris, 30debris, 50debris, 3digits, 4digits, 5digits or MNIST
"""
function recur_accuracy(recurrent_model, data_set, time_steps, dataset_name::String)
	acc = 0
	for (data, labels) in data_set
		# read the model output 1 times less, discard the output and read out again when calculating the onecold vector
		for i in 1:time_steps-1
			y_hat = recurrent_model(data)
		end
		y_hat = recurrent_model(data)
		Flux.reset!(recurrent_model)
		
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
"""
	ff_accuracy(feedforward_model, data_set, dataset_name::String)
Calculates the accuracy for a given feedforward model on the given dataset. The parameter dataset_name referes to the type of the dataset and can be one of the following:
10debris, 30debris, 50debris, 3digits, 4digits, 5digits or MNIST
"""
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
