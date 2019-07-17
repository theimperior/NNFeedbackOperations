include("/home/svendt/NNFeedbackOperations/Src/dataManager.jl")
using .dataManager: make_batch
using BSON: @load
using Printf
using Flux
using Flux: onecold
using CuArrays

struct dataset
	name::String
	idx::Int64 # for array indices (e.g. model_outputs)
	data::Array{Tuple{Array{Float32,4},Array{Float32,2}},1}
end

struct model
	name::String
	idx::Int64 # for array indices (e.g. model_outputs)
end

struct model_pair
	modelA::model
	modelB::model
end

mutable struct hypothesis
	models::model_pair
	dataset::Int64
	pval::Float64
	significance::Bool
	acceptance::Bool
end

pairwise_tests = [ 
	model_pair(1, 2), model_pair(1, 3), model_pair(1, 4), model_pair(1, 5), model_pair(1, 6), 
	model_pair(2, 3), model_pair(2, 4), model_pair(2, 5), model_pair(2, 6), 
	model_pair(3, 4), model_pair(3, 5), model_pair(3, 6), 
	model_pair(4, 5), model_pair(4, 6), 
	model_pair(5, 6)]

bool_str = ("FALSE", "TRUE")
const time_steps = 4
batch_size = 100

@info("loading datasets")
test_set_10debris, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitdebris/testset/mat/", ["5000_10debris1.mat", "5000_10debris2.mat"]..., batch_size=batch_size)
test_set_30debris, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitdebris/testset/mat/", ["5000_30debris1.mat", "5000_30debris2.mat"]..., batch_size=batch_size)
test_set_50debris, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitdebris/testset/mat/", ["5000_50debris1.mat", "5000_50debris2.mat"]..., batch_size=batch_size)
test_set_3digits, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitclutter/testset/mat/", ["5000_3digits1.mat", "5000_3digits2.mat"]..., batch_size=batch_size)
test_set_4digits, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitclutter/testset/mat/", ["5000_4digits1.mat", "5000_4digits2.mat"]..., batch_size=batch_size)
test_set_5digits, tmp1, tmp2 = make_batch("/home/svendt/NNFeedbackOperations/digitclutter/digitclutter/testset/mat/", ["5000_5digits1.mat", "5000_5digits2.mat"]..., batch_size=batch_size)

datasets = [dataset("10debris", 1, test_set_10debris), 	dataset("30debris", 2, test_set_30debris), 	dataset("50debris", 3, test_set_50debris), 
			dataset("3digits", 4, test_set_3digits), 	dataset("4digits", 5, test_set_4digits), 	dataset("5digits", 6, test_set_5digits)]
model_config = [model("BModel", 1), model("BKModel", 2), model("BFModel", 3), model("BLModel", 4), model"BTModel", 5), model("BLTModel", 6)]




# TODO move duplicate functions (-> with nets.jl) into separate module
function onematch(y::AbstractVector, targets::AbstractVector)
	if ( length(y) != length(targets) ) @warn("vectors in onematch(y::AbstractVector, targets::AbstractVector) differ in length, results may be unexpected!") end
	return targets[Base.argmax(y)]
end

function onekill(y::AbstractVector)
	y[Base.argmax(y)] = 0 
	return y
end

function onematch!(y::AbstractMatrix, targets::AbstractMatrix) 
	matches = dropdims(mapslices(x -> onematch(x[1:(length(x) รท 2)], x[(length(x) รท 2 + 1):length(x)]), vcat(y, targets), dims=1), dims=1)
	y[:, :] = mapslices(x -> onekill(x), y, dims=1)
	return matches
end
onematch!(y::TrackedMatrix, targets::AbstractMatrix) = onematch!(data(y), targets)

'''
function load_model(model_cfg::String, data_cfg::String)
	return "$(model_cfg)_$(data_cfg).bson"
end
'''
function load_model(m::model, d::dataset)
	@load "$(m.name)_$(d.name).bson" model acc
	return model
end
load_model(m::model) = return [load_model(m, d) for d in datasets]
@info("loading models")
models = [load_model(m) for m in model_config]


# generate model outputs
# returns the model output of _one_ model an _one_ batch
'''
get_FF_modeloutput(model, data::Array{Float32, 4}) = rand(Float32, 10, 100)
function get_FB_modeloutput(model, data::Array{Float32, 4}) = rand(Float32, 10, 100)
'''
get_FF_modeloutput(m, data::Array{Float32, 4}) = m(data)

function get_FB_modeloutput(m, data::Array{Float32, 4})
	y_hat = nothing
	for i in 1:time_steps
		y_hat = m(data)
	end
	Flux.reset!(m)
	return y_hat
end

# returns an array containing the output of _one_ model and _one_ dataset (e.g. 10debris)
function get_modeloutput(m, dataset::Array{Tuple{Array{Float32, 4}, Array{Float32, 2}}, 1}, model_cfg::model)
	if ( model_cfg.name == "BModel" || model_cfg.name  == "BKModel" || model_cfg.name  == "BFModel")
		return [get_FF_modeloutput(m, data) for (data, labels) in dataset]
	else 
		return [get_FB_modeloutput(m, data) for (data, labels) in dataset]
	end
end

# returns an array containing the output of _one_ model of _all_ datasets 
# Array{Array{Tuple{Array{Float32,4},Array{Float32,2}},1},1}
get_modeloutput(m, datasets::Array{dataset, 1}, model_cfg::model) where T <: Any = 
	return [get_modeloutput(m, datasets.data, model_cfg) for set in datasets]


#get_modeloutput(models::Array{Array{T,1}, 1}, datasets::Array{Array{Tuple{Tuple{Array{Float32, 2}, Array{Float32, 4}}, 1}, 1}) where T <: Any = 
#	return [get_modeloutput(models[model_cfg[2]], datasets, model_cfg) for model_cfg in model_config]

# typeof(model_outputs) = Array{Array{Array{Array{Float64,2},1},1},1}
# size(model_outputs) = (6,)
# size(model_outputs[1]) = (6,)
# size(model_outputs[1][1]) = (100,)
# size(model_outputs[1][1][1]) = (10, 100)
@info("retrieving model outputs")
model_outputs = [get_modeloutput(models[model_cfg.idx], datasets, model_cfg) for model_cfg in model_config]

function field_calculator(modelA_output, modelB_output, labels, num_targets)
	matchesA = zeros(size(labels[1,:]))
	matchesB = zeros(size(labels[1,:]))
	y_hatA = copy(modelA_output)
	y_hatB = copy(modelB_output)
	for it in 1:num_targets
		matchesA .+= onematch!(y_hatA, labels)
		matchesB .+= onematch!(y_hatB, labels)
	end
	
	a_ = min.(matchesA, matchesB)
	b_ = matchesB .- a_
	c_ = matchesA .- a_
	d_ = num_targets .- max.(matchesA, matchesB)
	return (sum(a_), sum(b_), sum(c_), sum(d_))
end

# returning the contingency table values for the pairwise McNemar test between model A and model B for a given dataset 
function pairwise_McNemar(modelA_output::Array{Array{Float32, 2}, 1}, modelB_output::Array{Array{Float32, 2}, 1}, dataset::dataset)
	fields = (0.0f0, 0.0f0, 0.0f0, 0.0f0) # a, b, c, d
	data_set = dataset.data
	if( dataset.name == "10debris" || dataset.name == "30debris" || dataset.name == "50debris" )
		for idx in 1:length(data_set)
			fields = fields .+ field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 1)
		end
	elseif ( dataset.name == "3digits" )
		for idx in 1:length(data_set)
			fields = fields .+ field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 3)
		end
	elseif ( dataset.name == "4digits" )
		for idx in 1:length(data_set)
			fields = fields .+ field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 4)
		end
	elseif ( dataset.name == "5digits" )
		for idx in 1:length(data_set)
			fields = fields .+ field_calculator(modelA_output[idx], modelB_output[idx], data_set[idx][2], 5)
		end
	end
	# return fields
	# calculation p-value and comparison with the chi-square distribution
	p_val = (fields[2] - fields[3])^2 / (fields[2] + fields[3])
	significance = false
	if ( p_val >= 3.84 ) 
		significance = true 
	end
	if ( sum(fields) != 10000 && sum(fields) != 30000 && sum(fields) != 40000 && sum(fields) != 50000 ) @warn("Fields adding up to a unexpected...") end
	return (fields, p_val, significance)
end

function pairwise_McNemar!(h::hypothesis)
	# map hypthesis to actual data and run McNemar test
	(fields, pval, sig) = pairwise_McNemar(model_outputs[hypothesis.models.modelA.idx][hypothesis.data_set.idx], model_outputs[hypothesis.models.modelA.idx][hypothesis.data_set.idx], hypothesis.data_set)
	hypothesis.p_val = pval
	hypothesis.significance = sig
end

# controlling the false positives using the Benjamini Hochberg procedure
# sorting the hypothesises according to their p-value and overwrites the statistical significance field according to Benjamini Hochberg
function FDR_control!(hypothesises, FDR::AbstractFloat)
	sort!(hypothesises, by = x -> x.p_val)
	m = length(hypothesises)
	rank = 1
	max_rank = 0
	for h in hypothesises
		if(h.p_val <= (rank/m) * FDR) max_rank = rank end
		rank += 1
	end
	
	# rejecting all null hypothesises for all H_i for i= 1...k 
	for idx in 1:max_rank
		# @printf("%s X = %f\n", p_values[idx][1], p_values[idx][2])
		hypothesises[idx].acceptance = false
	end
end

function print_results(hypothesises::Array{hypothesis, 1}, null_hypothesises::String)
	# null_hypothesises = copy(hypothesises)
	# sort hypothesises according to the statistical significance
	# sort!(null_hypothesises, by = x -> x[3])
	@printf("null hypothesis: %s \n", null_hypothesises)
	@printf("Dataset, ModelA, ModelB   |   p-value   |   statistical significance   |   accepting null hyp.\n")
	@printf("-------------------------------------------------------------------------------------------------\n")
	for h in hypothesises
		if ( h.significance && !h.acceptance ) marker = "<---" 
		else marker = ""
		end
		@printf("%s %s <-> %s  X = %f, statistical significant: %s, accepting null hypothesis: %s  %s\n", h.data_set.name, 
				h.models.modelA.name, h.models.modelB.name, h.p_val, bool_str[h.significance + 1], bool_str[h.acceptance + 1], marker)
	end
	@printf("\n\n\n")
end

# creating the null hypothesises
# hypothesises = null hypothesises of model A[1], model B[2] and dataset[3], p-value[2], bool for statistical significance and bool for accepting (true)/rejecting(false) null hypothesis
hypothesises = [] # typeof(hypothesises) = Array{Tuple{Tuple{Int64,Int64,Int64},Float64,Bool, Bool}, 1}
FDR = 0.05f0
for dataset in datasets
	for pairwise_test in pairwise_tests
		# defaulting to a not statistical significant hypothesis which would be accepted
		push!(hypothesises, hypothesis(pairwise_test, dataset, 0.0, false, true))
	end
	@info("run pairwise McNemar tests on Group")
	for h in hypothesises
		pairwise_McNemar!(h)
	end
	@info("controlling false positives at rate 0.05")
	FDR_control!(hypothesises, FDR)
	print_results(hypothesises, "statistical significant differences in model accuracy on one dataset")
	
	# run all tests in separate groups (divided by the different datasets)
	hypothesises = []
end









