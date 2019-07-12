include("./dataManager.jl")
using .dataManager: make_batch
using BSON: @load
using Flux
using CuArrays

const time_steps = 4
dataset_config = [("10debris", 1), ("30debris", 2), ("50debris", 3), ("3digits", 4), ("4digits", 5), ("5digits", 6)]
model_config = [("BModel", 1), ("BKModel", 2), ("BFModel", 3), ("BLModel", 4), ("BTModel", 5), ("BLTModel", 6)]
pairwise_tests = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]

# TODO do not concatenate!!!!!!
function get_FF_modeloutput(feedforward_model, test_data)
	model_output = nothing
	for (data, labels) in test_data
		if(isnothing(model_output))
			model_output = feedforward_model(data)
		else
			model_output = cat(dims=2, model_output, feedforward_model(data))
		end
	end
	return model_output
end

# TODO do not concatenate!!!!!!
function get_FB_modeloutput(reccurent_model, test_data)
	model_output = nothing
	for (data, labels) in data_set
		for i in 1:time_steps-1
			y_hat = reccurent_model(data)
		end
		if(isnothing(model_output))
			model_output = reccurent_model(data)
		else 
			model_output = cat(dims=2, model_output, reccurent_model(data))
		end
		Flux.reset!(reccurent_model)
	end
	return model_output
end

function pairwise_McNemar(modelA_output, modelB_output, data_set, config)
	# for digitdebris
	
	# for digitclutter 3
	# for digitclutter 4
	# for digitclutter 5
	
end

# load all testdatasets including the no debris dataset
# 15 McNemar tests for each dataset (6) = 90 tests

# have one array with all test datasets [[(),(),()], [(),(),()], [(),(),()], [(),(),()]]
# have one array with all modeloutputs for all datasets [[10x10000], [], [], []]

# load datasets
test_set_10debris, tmp1, tmp2 = make_batch("../digitclutter/digitdebris/testset/mat/", ["5000_10debris1.mat", "5000_10debris2.mat"]..., batch_size=batch_size)
test_set_30debris, tmp1, tmp2 = make_batch("../digitclutter/digitdebris/testset/mat/", ["5000_30debris1.mat", "5000_30debris2.mat"]..., batch_size=batch_size)
test_set_50debris, tmp1, tmp2 = make_batch("../digitclutter/digitdebris/testset/mat/", ["5000_50debris1.mat", "5000_50debris2.mat"]..., batch_size=batch_size)
test_set_3digits, tmp1, tmp2 = make_batch("../digitclutter/digitclutter/testset/mat/", ["5000_3digits1.mat", "5000_3digits2.mat"]..., batch_size=batch_size)
test_set_4digits, tmp1, tmp2 = make_batch("../digitclutter/digitclutter/testset/mat/", ["5000_4digits1.mat", "5000_4digits2.mat"]..., batch_size=batch_size)
test_set_5digits, tmp1, tmp2 = make_batch("../digitclutter/digitclutter/testset/mat/", ["5000_5digits1.mat", "5000_5digits2.mat"]..., batch_size=batch_size)

datasets = [test_set_10debris, test_set_30debris, test_set_50debris, test_set_3digits, test_set_4digits, test_set_5digits]

# load models (36 total)
models = Array{T}{undef, 6, 6}
for (model_cfg, m) in model_config
	for (data_cfg, i) in dataset_config
		@load "$model_cfg_$data_cfg.bson" model acc
		models[m, i] = model
	end
end

# generate model outputs
BModel_output = [get_FF_modeloutput(BModel[i], datasets[i]) for (cfg, i) in dataset_config]
BKModel_output = [get_FF_modeloutput(BKodel[i], datasets[i]) for (cfg, i) in dataset_config]
BFModel_output = [get_FF_modeloutput(BFodel[i], datasets[i]) for (cfg, i) in dataset_config]
BLModel_output = [get_FB_modeloutput(BLModel[i], datasets[i]) for (cfg, i) in dataset_config]
BTModel_output = [get_FB_modeloutput(BTModel[i], datasets[i]) for (cfg, i) in dataset_config]
BLTModel_output = [get_FB_modeloutput(BLTModel[i], datasets[i]) for (cfg, i) in dataset_config]
Model_output = [BModel_output, BKModel_output, BFModel_output, BLModel_output, BTModel_output, BLTModel_output]

# run pairwise McNemar tests
for (data_cfg, i) in dataset_config
	for (modelA, modelB) in pairwise_tests
		a, b, c, d = pairwise_McNemar(Model_output[modelA][i], Model_output[modelB][i], datasets[i] data_cfg)
		@show("...")
	end
end