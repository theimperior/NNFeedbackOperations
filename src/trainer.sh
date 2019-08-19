#!/bin/bash
CUDA_VISIBLE_DEVICES='1'
JULIA_DEBUG=Main
export CUDA_VISIBLE_DEVICES
export JULIA_DEBUG

julia nets03.jl
julia nets01.jl
julia nets003.jl
julia nets001.jl
julia nets0002.jl
julia nets0001.jl
julia nets00002.jl
julia nets00001.jl

