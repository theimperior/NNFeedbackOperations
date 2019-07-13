include("./dataManager.jl")
include("./nets.jl")
using .dataManager: make_batch
using .nets
using BSON: @load
using Flux
using Flux: onecold
using CuArrays
using Printf

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
@info("loading models")
models = [load_model(model_cfg) for model_cfg in model_config]



# load datasets
@info("loading datasets")
test_set_10debris, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitdebris/testset/mat/", ["5000_10debris1.mat", "5000_10debris2.mat"]..., batch_size=batch_size)
test_set_30debris, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitdebris/testset/mat/", ["5000_30debris1.mat", "5000_30debris2.mat"]..., batch_size=batch_size)
test_set_50debris, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitdebris/testset/mat/", ["5000_50debris1.mat", "5000_50debris2.mat"]..., batch_size=batch_size)
test_set_3digits, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitclutter/testset/mat/", ["5000_3digits1.mat", "5000_3digits2.mat"]..., batch_size=batch_size)
test_set_4digits, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitclutter/testset/mat/", ["5000_4digits1.mat", "5000_4digits2.mat"]..., batch_size=batch_size)
test_set_5digits, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitclutter/testset/mat/", ["5000_5digits1.mat", "5000_5digits2.mat"]..., batch_size=batch_size)
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
@info("retrieving model outputs")
model_outputs = [get_modeloutput(models[model_cfg[2]], datasets, model_cfg) for model_cfg in model_config]
# generate model outputs
# BModel_output = [get_FF_modeloutput(BModel[i], datasets[i]) for (cfg, i) in dataset_config]
# BKModel_output = [get_FF_modeloutput(BKodel[i], datasets[i]) for (cfg, i) in dataset_config]
# BFModel_output = [get_FF_modeloutput(BFodel[i], datasets[i]) for (cfg, i) in dataset_config]
# BLModel_output = [get_FB_modeloutput(BLModel[i], datasets[i]) for (cfg, i) in dataset_config]
# BTModel_output = [get_FB_modeloutput(BTModel[i], datasets[i]) for (cfg, i) in dataset_config]
# BLTModel_output = [get_FB_modeloutput(BLTModel[i], datasets[i]) for (cfg, i) in dataset_config]
# Model_output = [BModel_output, BKModel_output, BFModel_output, BLModel_output, BTModel_output, BLTModel_output]
# model_output = [get_modeloutput(models[m, :], datasets) for (model_cfg, m) in model_config]

function field_calculator(modelA_output, modelB_output, labels, num_targets)
	matchesA = zeros(size(y_hat[1,:]))
	matchesB = zeros(size(y_hat[1,:]))
	y_hat = copy(modelA_output)
	y_hat = copy(modelB_output)
	for it in 1:num_targets
		matchesA .+= onematch!(y_hat, labels)
		matchesB .+= onematch!(y_hat, labels)
	end
	
	a_ = min.(matchesA, matchesB)
	b_ = matchesB .- a_
	c_ = matchesA .- b_
	d_ = num_targets .- max.(matchesA, matchesB)
	return (sum(a_), sum(b_), sum(c_), sum(d_))
end

# returning the contingency table values for the pairwise McNemar test between model A and model B for a given dataset 
function pairwise_McNemar(modelA_output::Array{Array{Float32, 2}, 1}, modelB_output::Array{Array{Float32, 2}, 1}, data_set, dataset_cfg::String)
	fields = (0, 0, 0, 0) # a, b, c, d
	if( dataset_cfg == "10debris" || dataset_cfg == "30debris" || dataset_cfg == "50debris" )
		for idx in 1:length(data_set)
			# Model A correct & Model B correct
			# a += count(onecold(modelA_output[idx]) .== onecold(data_set[idx][2]) .& onecold(modelB_output[idx]) .== onecold(data_set[idx][2]))
			# Model A incorrect & Model B correct
			# b += count(.~(onecold(modelA_output[idx]) .== onecold(data_set[idx][2])) .& onecold(modelB_output[idx]) .== onecold(data_set[idx][2]))
			# Model A correct & Model B incorrect
			# c += count(onecold(modelA_output[idx]) .== onecold(data_set[idx][2]) .& .~(onecold(modelB_output[idx]) .== onecold(data_set[idx][2])))
			# Model A incorrect & Model B incorrect
			# d += count(.~(onecold(modelA_output[idx]) .== onecold(data_set[idx][2])) .& .~(onecold(modelB_output[idx]) .== onecold(data_set[idx][2])))
			fields .+= field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 1)
		end
	elseif ( dataset_cfg == "3digits" )
		for idx in 1:length(data_set)
			fields .+= field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 3)
		end
	elseif ( dataset_cfg == "4digits" )
		for idx in 1:length(data_set)
			fields .+= field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 4)
		end
	elseif ( dataset_cfg == "5digits" )
		for idx in 1:length(data_set)
			fields .+= field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 5)
		end
	end
	return (a, b, c, d)
end

# load all testdatasets including the no debris dataset
# 15 McNemar tests for each dataset (6) = 90 tests

# have one array with all test datasets [[(),(),()], [(),(),()], [(),(),()], [(),(),()]]
# have one array with all modeloutputs for all datasets [[10x10000], [], [], []]



# run pairwise McNemar tests
@info("run pairwise McNemar tests")
for (data_cfg, i) in dataset_config
	for (modelA, modelB) in pairwise_tests
		fields = pairwise_McNemar(model_outputs[modelA][i], model_outputs[modelB][i], datasets[i], data_cfg)
		p_val = (fields[2] - fields[3])^2 / (fields[2] + fields[3])
		@printf("%s %s <-> %s  X(1, N = %d) = %f", data_cfg, model_config[modelA][1], model_config[modelB][1], sum(fields), p_val)
	end
end




