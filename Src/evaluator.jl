include("./dataManager.jl")
using .dataManager: make_batch
using BSON: @load
using Flux
using Flux: onecold
using CuArrays

const time_steps = 4
dataset_config = [("10debris", 1), ("30debris", 2), ("50debris", 3), ("3digits", 4), ("4digits", 5), ("5digits", 6)]
model_config = [("BModel", 1), ("BKModel", 2), ("BFModel", 3), ("BLModel", 4), ("BTModel", 5), ("BLTModel", 6)]
pairwise_tests = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]

""" structure of the data
 model_output
  _
 |
 | ...tbc
 |_
 
""" 

# load models (36 total)
function load_model(model_cfg::String, data_cfg::String)
	@load "$model_cfg_$data_cfg.bson" model acc
	return model
end
load_model(model_cfg::Tuple{String, Int64}) = return [load_model(model_cfg[1], dataset_cfg) for (dataset_cfg, i) in dataset_config]
models = [load_model(model_cfg) for model_cfg in model_config]



# load datasets
test_set_10debris, tmp1, tmp2 = make_batch("../digitclutter/digitdebris/testset/mat/", ["5000_10debris1.mat", "5000_10debris2.mat"]..., batch_size=batch_size)
test_set_30debris, tmp1, tmp2 = make_batch("../digitclutter/digitdebris/testset/mat/", ["5000_30debris1.mat", "5000_30debris2.mat"]..., batch_size=batch_size)
test_set_50debris, tmp1, tmp2 = make_batch("../digitclutter/digitdebris/testset/mat/", ["5000_50debris1.mat", "5000_50debris2.mat"]..., batch_size=batch_size)
test_set_3digits, tmp1, tmp2 = make_batch("../digitclutter/digitclutter/testset/mat/", ["5000_3digits1.mat", "5000_3digits2.mat"]..., batch_size=batch_size)
test_set_4digits, tmp1, tmp2 = make_batch("../digitclutter/digitclutter/testset/mat/", ["5000_4digits1.mat", "5000_4digits2.mat"]..., batch_size=batch_size)
test_set_5digits, tmp1, tmp2 = make_batch("../digitclutter/digitclutter/testset/mat/", ["5000_5digits1.mat", "5000_5digits2.mat"]..., batch_size=batch_size)
datasets = [test_set_10debris, test_set_30debris, test_set_50debris, test_set_3digits, test_set_4digits, test_set_5digits]

# generate model outputs
# returns the model output of _one_ model an _one_ batch
get_FF_modeloutput(model, data::Array{Float32, 4}) = model(data)

function get_FB_batch_modeloutput(model, data::Array{Float32, 4})
	y_hat = nothing
	for i in 1:time_steps
		y_hat = model(data)
	end
	Flux.reset!(reccurent_model)
	return y_hat
end

# returns an array containing the output of _one_ model and _one_ dataset (e.g. 10debris)
function get_modeloutput(model, dataset::Array{Tuple{Array{Float32, 2}, Array{Float32, 4}}, 1}, model_cfg::String)
	if ( model_cfg == "BModel" || model_cfg == "BKModel" || model_cfg == "BFModel")
		return [get_FF_batch_modeloutput(model, data) for (data, labels) in data_set]
	else 
		return [get_FB_batch_modeloutput(model, data) for (data, labels) in data_set]
	end
end

# returns an array containing the output of _one_ model of _all_ datasets
get_modeloutput(model::Array{T, 1}, datasets::Array{Array{Tuple{Tuple{Array{Float32, 2}, Array{Float32, 4}}, 1}, 1}, model_cfg::Tuple{String, Int64}) where T <: Any = 
	return [get_modeloutput(model[i], datasets[i], model_cfg[2]) for (dataset_cfg, i) in dataset_config]


#get_modeloutput(models::Array{Array{T,1}, 1}, datasets::Array{Array{Tuple{Tuple{Array{Float32, 2}, Array{Float32, 4}}, 1}, 1}) where T <: Any = 
#	return [get_modeloutput(models[model_cfg[2]], datasets, model_cfg) for model_cfg in model_config]
model_outputs = [get_modeloutput(models[model_cfg[2]], datasets, model_cfg) for model_cfg in model_config]



# easier would be McNemar on modelA, modelB and a dataset and then generate the modeloutput within mcnemar
# returning the p_value for the pairwise McNemar test between model A and model B for a given dataset 
function pairwise_McNemar(modelA_output::Array{Array{Float32, 2}, 1}, modelB_output::Array{Array{Float32, 2}, 1}, data_set, dataset_cfg::String)
	# for digitdebris
	a = 0
	b = 0
	c = 0
	d = 0
	if( dataset_cfg == "10debris" || dataset_cfg == "30debris" || dataset_cfg == "50debris" )
		for idx in 1:length(data_set)
			a += count(onecold(modelA_output[idx]) .== onecold(data_set[idx][2]) && onecold(modelB_output[idx]) .== onecold(data_set[idx][2]))
			b += 
			c += 
			d += 
		end
	elseif ( dataset_cfg == "3digits" )
		
	elseif ( dataset_cfg == "4digits" )
	elseif ( dataset_cfg == "5digits" )
	end
	# for digitclutter 3
	# for digitclutter 4
	# for digitclutter 5
end

# load all testdatasets including the no debris dataset
# 15 McNemar tests for each dataset (6) = 90 tests

# have one array with all test datasets [[(),(),()], [(),(),()], [(),(),()], [(),(),()]]
# have one array with all modeloutputs for all datasets [[10x10000], [], [], []]





# generate model outputs
# BModel_output = [get_FF_modeloutput(BModel[i], datasets[i]) for (cfg, i) in dataset_config]
# BKModel_output = [get_FF_modeloutput(BKodel[i], datasets[i]) for (cfg, i) in dataset_config]
# BFModel_output = [get_FF_modeloutput(BFodel[i], datasets[i]) for (cfg, i) in dataset_config]
# BLModel_output = [get_FB_modeloutput(BLModel[i], datasets[i]) for (cfg, i) in dataset_config]
# BTModel_output = [get_FB_modeloutput(BTModel[i], datasets[i]) for (cfg, i) in dataset_config]
# BLTModel_output = [get_FB_modeloutput(BLTModel[i], datasets[i]) for (cfg, i) in dataset_config]
# Model_output = [BModel_output, BKModel_output, BFModel_output, BLModel_output, BTModel_output, BLTModel_output]
# model_output = [get_modeloutput(models[m, :], datasets) for (model_cfg, m) in model_config]


# run pairwise McNemar tests
for (data_cfg, i) in dataset_config
	for (modelA, modelB) in pairwise_tests
		a, b, c, d = pairwise_McNemar(Model_output[modelA][i], Model_output[modelB][i], datasets[i] data_cfg)
		@show("...")
	end
end