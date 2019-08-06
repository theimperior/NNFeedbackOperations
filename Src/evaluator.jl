include("/home/svendt/NNFeedbackOperations/Src/dataManager.jl")
using .dataManager: make_batch
using BSON: @load
using Printf
using Flux
using Flux: onecold
using CuArrays
using DataFrames
using GLM
using Random

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
	dataset::dataset
	pval::Float64
	significance::Bool
	acceptance::Bool
end

model_config = [model("BModel  ", 1), model("BKModel ", 2), model("BFModel ", 3), model("BLModel ", 4), model("BTModel ", 5), model("BLTModel", 6)]

pairwise_tests = [ 
	model_pair(model_config[1], model_config[2]), model_pair(model_config[1], model_config[3]), model_pair(model_config[1], model_config[4]), model_pair(model_config[1], model_config[5]), model_pair(model_config[1], model_config[6]), 
	model_pair(model_config[2], model_config[3]), model_pair(model_config[2], model_config[4]), model_pair(model_config[2], model_config[5]), model_pair(model_config[2], model_config[6]), 
	model_pair(model_config[3], model_config[4]), model_pair(model_config[3], model_config[5]), model_pair(model_config[3], model_config[6]), 
	model_pair(model_config[4], model_config[5]), model_pair(model_config[4], model_config[6]), 
	model_pair(model_config[5], model_config[6])]

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
			dataset("3digits ", 4, test_set_3digits), 	dataset("4digits ", 5, test_set_4digits), 	dataset("5digits ", 6, test_set_5digits)]





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
	matches = dropdims(mapslices(x -> onematch(x[1:(length(x) ÷ 2)], x[(length(x) ÷ 2 + 1):length(x)]), vcat(y, targets), dims=1), dims=1)
	y[:, :] = mapslices(x -> onekill(x), y, dims=1)
	return matches
end
onematch!(y::TrackedMatrix, targets::AbstractMatrix) = onematch!(data(y), targets)

'''
function load_model(m::model, d::dataset)
	return "$(m.name)_$(d.name).bson"
end
'''
function load_model(m::model, d::dataset)
	@load "$(m.name)_$(d.name).bson" model acc
	return model
end
load_model(m::model) = return [load_model(m, d) for d in datasets]
@info("loading models")
models = [load_model(m) for m in model_config]
'''6-element Array{Array{String,1},1}:
 ["BModel  _10debris.bson", "BModel  _30debris.bson", "BModel  _50debris.bson", "BModel  _3digits.bson", "BModel  _4digits.bson", "BModel  _5digits.bson"]
 ["BKModel _10debris.bson", "BKModel _30debris.bson", "BKModel _50debris.bson", "BKModel _3digits.bson", "BKModel _4digits.bson", "BKModel _5digits.bson"]
 ["BFModel _10debris.bson", "BFModel _30debris.bson", "BFModel _50debris.bson", "BFModel _3digits.bson", "BFModel _4digits.bson", "BFModel _5digits.bson"]
 ["BLModel _10debris.bson", "BLModel _30debris.bson", "BLModel _50debris.bson", "BLModel _3digits.bson", "BLModel _4digits.bson", "BLModel _5digits.bson"]
 ["BTModel _10debris.bson", "BTModel _30debris.bson", "BTModel _50debris.bson", "BTModel _3digits.bson", "BTModel _4digits.bson", "BTModel _5digits.bson"]
 ["BLTModel_10debris.bson", "BLTModel_30debris.bson", "BLTModel_50debris.bson", "BLTModel_3digits.bson", "BLTModel_4digits.bson", "BLTModel_5digits.bson"]
'''

function print_accs(accs::Array{Array{Float64,1},1})
	@printf("Final accuracies on testsets for all models and all datasets\n")
	# model name 8
	# dataset name 8 
	@printf("  model   | dataset  |  accuracy / error \n")
	@printf("------------------------------------------\n")
	for model_cfg in model_config
		for dataset in datasets
			if ( dataset.idx != 3 ) @printf("          | %s | %f / %f\n", dataset.name, accs[model_cfg.idx][dataset.idx], 1 - accs[model_cfg.idx][dataset.idx])
			else @printf(" %s | %s | %f / %f\n", model_cfg.name, dataset.name, accs[model_cfg.idx][dataset.idx], 1 - accs[model_cfg.idx][dataset.idx])
			end
		end
		@printf("------------------------------------------\n")
	end
end

function load_accuracy(m::model, d::dataset)
	@load "$(m.name)_$(d.name).bson" model acc
	return acc
end
load_accuracy(m::model) = return [load_accuracy(m, d) for d in datasets]
@info("loading accuracy")
accs = [load_accuracy(m) for m in model_config]
print_accs(accs)
'''
6-element Array{Array{Float64,1},1}
'''

# generate model outputs
# returns the model output of _one_ model an _one_ batch
'''
get_FF_modeloutput(model, data::Array{Float32, 4}) = rand(Float32, 10, 100)
get_FB_modeloutput(model, data::Array{Float32, 4}) = rand(Float32, 10, 100)
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
function get_modeloutput(m::model, set::dataset)
	if ( m.name == "BModel" || m.name  == "BKModel" || m.name  == "BFModel")
		return [get_FF_modeloutput(models[m.idx][set.idx], data) for (data, labels) in set.data]
	else 
		return [get_FB_modeloutput(models[m.idx][set.idx], data) for (data, labels) in set.data]
	end
end

# returns an array containing the output of _one_ model of _all_ datasets 
get_modeloutput(m::model) =
	return [get_modeloutput(m, set) for set in datasets]

# typeof(model_outputs) = Array{Array{Array{Array{Float64,2},1},1},1}
# size(model_outputs) = (6,)
# size(model_outputs[1]) = (6,)
# size(model_outputs[1][1]) = (100,)
# size(model_outputs[1][1][1]) = (10, 100)
@info("retrieving model outputs")
model_outputs = [get_modeloutput(model_cfg) for model_cfg in model_config]

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
	(fields, pval, sig) = pairwise_McNemar(model_outputs[h.models.modelA.idx][h.dataset.idx], 
											model_outputs[h.models.modelB.idx][h.dataset.idx], 
											h.dataset)
	h.pval = pval
	h.significance = sig
end

# controlling the false positives using the Benjamini Hochberg procedure
# sorting the hypothesises according to their p-value and overwrites the statistical significance field according to Benjamini Hochberg
function FDR_control!(hypothesises, FDR::AbstractFloat)
	sort!(hypothesises, by = x -> x.pval)
	m = length(hypothesises)
	rank = 1
	max_rank = 0
	for h in hypothesises
		if(h.pval <= (rank/m) * FDR) max_rank = rank end
		rank += 1
	end
	
	# rejecting all null hypothesises for all H_i for i= 1...k 
	for idx in 1:max_rank
		# @printf("%s X = %f\n", p_values[idx][1], p_values[idx][2])
		hypothesises[idx].acceptance = false
	end
end

function print_results(hypothesises::Array{Any, 1}, null_hypothesises::String)
	@printf("null hypothesis: %s \n", null_hypothesises)
	@printf("Dataset, ModelA, ModelB   |   p-value   |   statistical significance      |   accepting null hyp.\n")
	@printf("-------------------------------------------------------------------------------------------------\n")
	for h in hypothesises
		if ( h.significance && !h.acceptance ) marker = "<---" 
		else marker = ""
		end
		@printf("%s %s <-> %s  X = %f, statistical significant: %s, accepting null hypothesis: %s  %s\n", h.dataset.name, 
				h.models.modelA.name, h.models.modelB.name, h.pval, bool_str[h.significance + 1], bool_str[h.acceptance + 1], marker)
	end
	@printf("\n\n\n")
end


# hypothesises = null hypothesises of model A[1], model B[2] and dataset[3], p-value[2], bool for statistical significance and bool for accepting (true)/rejecting(false) null hypothesis
nh = [] # null hypothesises
FDR = 0.05f0
for dataset in datasets
	for pairwise_test in pairwise_tests
		# defaulting to a not statistical significant hypothesis which would be accepted
		push!(nh, hypothesis(pairwise_test, dataset, 0.0, false, true))
	end
	@info("run pairwise McNemar tests on Group $(dataset.name), FDR at $(FDR)")
	for h in nh
		pairwise_McNemar!(h)
	end
	@info("controlling false positives at rate 0.05")
	FDR_control!(nh, FDR)
	print_results(nh, "statistical significant diff in model accuracy on dataset $(dataset.name)")
	
	# run all tests in separate groups (divided by the different datasets)
	nh = []
end


# store accuarcy somewhere 
# construct null distribution
	# assuming to fit a gaussian 
	
#construct null distributions
function shuffle!(a::Array{Float32,2}, b::Array{Float32,2}) 
	perm = bitrand(size(a, 2))
	tmp_a = similar(a[:, 1])
	for idx in 1:length(perm)
		if(perm[idx]) 
			tmp_a = a[:, idx]
			a[:, idx] = b[:, idx]
			b[:, idx] = tmp_a
		end
	end
end

shuffle!(a::Array{Array{Float32,2},1}, b::Array{Array{Float32,2},1}) = 	
	for idx in 1:size(a, 1)
		shuffle!(a[idx], b[idx])
	end
reps_null_distro = 10000
X = [1, 2, 3]
Y_A = [0, 0, 0]
Y_B = [0, 0, 0]

for model_pair in pairwise_tests
	# null distribution for digit __debris__
	
	# collect results (10debris, 30debris, 50debris) so random shuffling will be easier
	model_outputA = [model_outputs[model_pair.modelA.idx][1], 
						model_outputs[model_pair.modelA.idx][2], 
						model_outputs[model_pair.modelA.idx][3]]
	model_outputB = [model_outputs[model_pair.modelB.idx][1], 
						model_outputs[model_pair.modelB.idx][2], 
						model_outputs[model_pair.modelB.idx][3]]
	diffs = zeros(reps_null_distro)
	for perm in 1:reps_null_distro
		model_outputA_perm = copy(model_outputA)
		model_outputB_perm = copy(model_outputB)
		# shuffle predictions of single images between model pair for all three datasets
		for idx in 1:3
			shuffle!(model_outputA_perm[idx], model_outputB_perm[idx])
			Y_A[idx] = accuracy(model_outputA_perm, labels) # TODO create function
			Y_B[idx] = accuracy(model_outputB_perm, labels) # TODO create function
		end
		# fit linear models 
		lmA = lm(@formula(Y ~ X), DataFrame(X=X, Y=Y_A))
		lmB = lm(@formula(Y ~ X), DataFrame(X=X, Y=Y_B))
		paramsA = round.(coef(lmA), digits=5)
		paramsB = round.(coef(lmB), digits=5)
		
		diff = paramsA .- paramsB
		diffs[perm] = diff[1] + diff[2]
	end
	# fit a normal distribution to the differences
	null_dist = fit(Normal, diffs)
	
	# TODO calculate real differences
	# cumulative propability cdf(distribution, value)
	# p_val = pdf(null_dist, real_diff)
	# TODO Check weather p_val 
end





