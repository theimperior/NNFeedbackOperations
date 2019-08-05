module accuracy

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
	matches = dropdims(mapslices(x -> onematch(x[1:(length(x) รท 2)], x[(length(x) รท 2 + 1):length(x)]), vcat(y, targets), dims=1), dims=1)
	y[:, :] = mapslices(x -> onekill(x), y, dims=1)
	return matches
end

onematch!(y::TrackedMatrix, targets::AbstractMatrix) = onematch!(Tracker.data(y), targets)

function recur_accuracy(reccurent_model, data_set, config::String)
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
	elseif ( config == "3digits" )
		for (data, labels) in data_set
			for i in 1:time_steps-1
				y_hat = reccurent_model(data)
			end
			y_hat = reccurent_model(data)
			# This has been tested and is executed one after the other
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 3)
			Flux.reset!(reccurent_model)
		end
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
	else
	end
	return acc / length(data_set)
end

function ff_accuracy(feedforward_model, data_set, config::String)
	acc = 0
	if( config == "10debris" || config == "30debris" || config == "50debris" )
		for (data, labels) in data_set
			acc += mean(onecold(feedforward_model(data)) .== onecold(labels))
		end
	elseif ( config == "3digits" )
		for (data, labels) in data_set
			y_hat = feedforward_model(data)
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 3)
		end
	elseif ( config == "4digits" )
		for (data, labels) in data_set
			y_hat = feedforward_model(data)
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 4)
		end
	elseif ( config == "5digits" )
		for (data, labels) in data_set
			y_hat = feedforward_model(data)
			matches = onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels) .+ onematch!(y_hat, labels)
			acc += mean(matches .== 5)
		end
	else
	end
	return acc / length(data_set)
end

end # module accuracy