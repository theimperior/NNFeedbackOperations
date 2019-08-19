#!/bin/bash
CUDA_VISIBLE_DEVICES='1'
JULIA_DEBUG=Main
export CUDA_VISIBLE_DEVICES
export JULIA_DEBUG

epochs=35
dataset="MNIST"

julia nets.jl 0.03 $epochs $dataset "" BLModel
julia nets.jl 0.01 $epochs $dataset "" BLModel
julia nets.jl 0.002 $epochs $dataset "" BLModel
julia nets.jl 0.03 $epochs $dataset "" BTModel
julia nets.jl 0.01 $epochs $dataset "" BTModel
julia nets.jl 0.03 $epochs $dataset "" BLTModel
julia nets.jl 0.01 $epochs $dataset "" BLTModel
julia nets.jl 0.002 $epochs $dataset "" BLTModel

