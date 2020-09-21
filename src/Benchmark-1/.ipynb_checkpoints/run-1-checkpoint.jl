using JLD2
using FileIO
using PyPlot
using Statistics
using StatsBase 
using Distributions
using LinearAlgebra
using BAT
using IntervalSets
using Random, ArraysOfArrays
using ValueShapes
using Measurements
using TypedTables
using HDF5
using CSV

include("../likelihood.jl")
include("../utill.jl")

images = load("../../data/experiment/dataset_2/m2/simulated_events.jld2")["array"];
conv_mat = load("../../data/experiment/dataset_2/m2/conv_matrix.jld2")

name = "../../data/sampling_results/Benchmark-1/sim_event-1-"

conv_matrices = (
    cam_1 = conv_mat["cam_1"],
    cam_2 = conv_mat["cam_2"],
    cam_3 = conv_mat["cam_3"],
    cam_4 = conv_mat["cam_3"],
)

param_truth = (
        tr_size = [0.15, 0.15],
        ang_spr = [4.0, 4.0],
        s_waist = [2.9,],
        μ_x = [35,35,20,35], 
        μ_y = [35,35,20,35], 
        σ_x = [0,0,0,0], 
        σ_y = [0,0,0,0], 
        δ_x = [0.0271,0.0216,0.114,3*0.0303], 
        δ_y = [0.0305,0.0234,0.125,3*0.0298],
        int_coeff  = [35147.44, 50235.06, 10096.64, 33406.9],
        s_cam = [0.0, 1.478, 15.026, 23.1150], 
    )

tuning = AdaptiveMetropolisTuning(
    λ = 0.5,
    α = 0.15..0.25,
    β = 1.5,
    c = 1e-4..1e2,
    r = 0.5,
)

convergence = BrooksGelmanConvergence(
    threshold = 1.1,
    corrected = false
)

init = MCMCInitStrategy(
    init_tries_per_chain = 100..200,
    max_nsamples_init = 600,
    max_nsteps_init = 600,
    max_time_init = Inf
)

burnin = MCMCBurninStrategy(
    max_nsamples_per_cycle = 1700,
    max_nsteps_per_cycle = 1700,
    max_time_per_cycle = Inf,
    max_ncycles = 100
);

algorithm = MetropolisHastings()

β_min = 0.8
β_max = 1.2

prior = NamedTupleDist(
        tr_size = [β_min*param_truth.tr_size[1]..β_max*param_truth.tr_size[1], β_min*param_truth.tr_size[2]..β_max*param_truth.tr_size[2]],
        ang_spr = [β_min*param_truth.ang_spr[1]..β_max*param_truth.ang_spr[1], β_min*param_truth.ang_spr[2]..β_max*param_truth.ang_spr[2]],
        s_waist = [β_min*param_truth.s_waist[1]..β_max*param_truth.s_waist[1],],
        μ_x = [35,35,20,35], 
        μ_y = [35,35,20,35], 
        σ_x = [0,0,0,0], 
        σ_y = [0,0,0,0], 
        δ_x = [0.0271,0.0216,0.114,3*0.0303], 
        δ_y = [0.0305,0.0234,0.125,3*0.0298],
        int_coeff  = [35147.44, 50235.06, 10096.64, 33406.9],
        s_cam = [0.0, 1.478, 15.026, 23.1150], 
    );

nsamples = 6*10^4
nchains = 4

global table = 0.0

for (ind, event) in enumerate(images)
    
    @show ind

    log_likelihood = let e = event, c = conv_matrices

        params -> begin

            ll = zero(Float64)
            ll += cam_likelihood(params, e.cam_1, e.population, c.cam_1, 1)
            ll += cam_likelihood(params, e.cam_2, e.population, c.cam_2, 2)
            ll += cam_likelihood(params, e.cam_3, e.population, c.cam_3, 3)
            ll += cam_likelihood(params, e.cam_4, e.population, c.cam_4, 4)

            return LogDVal(ll)

        end
    end

    posterior = PosteriorDensity(log_likelihood, prior);
    
    
    el_time = @elapsed samples = bat_sample(
        posterior, (nsamples, nchains), algorithm,
        max_nsteps = nsamples,
        max_time = Inf,
        tuning = tuning,
        init = init,
        burnin = burnin,
        convergence = convergence,
        strict = true,
        filter = true).result
    
    BAT.bat_write(name*"$ind.hdf5", unshaped.(samples))
    
    if ind == 1 
        global table = TypedTables.Table(
            time = [el_time], 
            name=[name[end-11:end]*"$ind"])
    else
        table_tmp = TypedTables.Table(
            time = [el_time], 
            name=[name[end-11:end]*"$ind"]
        )
        append!(table, table_tmp)
    end   
    
end

CSV.write(name*"table.csv", table)