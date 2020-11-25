using JLD2
using FileIO
using Statistics
using StatsBase 
using Distributions
using LinearAlgebra
using HDF5
using BenchmarkTools
using IntervalSets
using Random, ArraysOfArrays
using ValueShapes
using Measurements
using BenchmarkTools
using BAT 

include("../model-41/likelihood.jl")

function def_conv_mat()
    conv_mat = load("../../data/experiment/dataset_2/m1/conv-matrix-upd-2.jld2")
    return (cam_1 = conv_mat["cam_1"], cam_2 = conv_mat["cam_2"], cam_3 = conv_mat["cam_3"], cam_4 = conv_mat["cam_4"]) 
end

function def_data_vector(ev_ind)
    images = load("../../data/experiment/dataset_2/m1/images-satur.jld2")
    
    return [( cam_1 = images["cam_1"][i ,:,:], cam_2 = images["cam_2"][i ,:,:], 
            cam_3 = images["cam_3"][i ,:,:], cam_4 = images["cam_4"][i ,:,:], 
            population = images["charge"][i ,:][1],) for i in ev_ind]
end

function def_rem_ind()
    images = load("../../data/experiment/dataset_2/m1/images-satur.jld2")
    ind_tmp = [1, 137, 151, 169, 2, 225, 262, 299, 3, 343, 355, 369, 4, 48, 5, 6, 7, 80]
    rem_ind = setdiff(eachindex(images["charge"]), ind_tmp)
    return rem_ind
end

function log_lik_ndiff(e, cv_mat; 
        func = conv_tabl_discrete,
        n_threads = Threads.nthreads(),
    )
    
    cv_1 = cv_mat.cam_1
    cv_2 = cv_mat.cam_2
    cv_3 = cv_mat.cam_3
    
    return params -> begin 
        ll = 0.0
        ll += likelihood_cam13(params, e.cam_1, e.population, cv_1, func, 1, n_threads=n_threads)
        ll += likelihood_cam13(params, e.cam_2, e.population, cv_2, func, 2, n_threads=n_threads)
        ll += likelihood_cam13(params, e.cam_3, e.population, cv_3, func, 3, n_threads=n_threads)
        ll += likelihood_cam4(params, e.cam_4, e.population, 4, n_threads=n_threads)
        return LogDVal(ll)
    end
        
end

function def_settings()
    
    tuning = AdaptiveMHTuning(
        λ = 0.5,
        α = ClosedInterval(0.15,0.25),
        β = 1.5,
        c = ClosedInterval(1e-4,1e2),
        r = 0.5,
    )

    convergence = BrooksGelmanConvergence(
        threshold = 1.1,
        corrected = false
    )

    init = MCMCChainPoolInit(
        init_tries_per_chain = ClosedInterval(50,150),
        max_nsamples_init = 500,
        max_nsteps_init = 500,
        max_time_init = Inf
    )

    burnin = MCMCMultiCycleBurnin(
        max_nsamples_per_cycle = 12000,
        max_nsteps_per_cycle = 12000,
        max_time_per_cycle = Inf,
        max_ncycles = 130
    )
    
    return tuning, convergence, init, burnin 
end

function def_prior()
    β1= 0.015
    β2 = 0.0077
    β3 = 0.0058 

    return   NamedTupleDist(
        tr_size = [truncated(Normal(0.2, 0.04), 0.02, 0.16), truncated(Normal(0.2, 0.04), 0.02, 0.16)],
        tr_size_2 = [truncated(Normal(0.2, 0.04), 0.02, 0.16), truncated(Normal(0.2, 0.04), 0.02, 0.16)],
        ang_spr = [truncated(Normal(4.0, 2.0), 4.0, 8.5), truncated(Normal(4.0, 2.0), 4.0, 8.5)],
        ang_spr_2 = [truncated(Normal(4.0, 2.0), 1.0, 4.0), truncated(Normal(4.0, 2.0), 1.0, 4.0)],
        mixt_pow =  0.50 .. 1.0 ,
        waist = [truncated(Normal(2.9, 0.03), 2.65, 3.5)],
        waist_2 = [truncated(Normal(2.9, 0.03), 2.65, 3.5)], # 11
        algmx = [23.0 .. 48, 23.0 .. 48.0, 10.0 .. 30.0, 23.0 .. 48.0],
        algmy = [23.0 .. 48, 23.0 .. 48.0, 10.0 .. 30.0, 23.0 .. 48.0],
        cam4_ped = 4.0 .. 40.0,
        cam4_light_fluct =  1.0 .. 3.0,
        cam4_light_amp = 1.6 .. 9.9, 
        resx = [truncated(Normal(1, 0.5), 0, Inf), truncated(Normal(1, 0.5), 0, Inf), truncated(Normal(1, 0.5), 0, Inf)], 
        resy = [truncated(Normal(1, 0.5), 0, Inf), truncated(Normal(1, 0.5), 0, Inf), truncated(Normal(1, 0.5), 0, Inf)], 
        cam4_resx = truncated(Normal(3, 1.5), 0, Inf),
        cam4_resy = truncated(Normal(3, 1.5), 0, Inf), 
        psx = [27.1, 21.6, 114.0], # 31, 32, 33
        psy = [30.5, 23.4, 125.0], # 34, 35, 36
        cam4_psx = 121.8, # 37
        cam4_psy = 120.0, # 38
        light_amp  = [1.0 .. 13.0 , 1.0 .. 17.0, 1.0 .. 5.0], # 1.0 .. 5.0
        s_cam = [0.0, 1.478, 15.026, 23.1150],
    )
end

function main(event_ind)
    
    prior = def_prior()
    data = def_data_vector(event_ind)
    conv_mat = def_conv_mat()
    tuning, convergence, init, burnin  = def_settings()
    nsamples, nchains = 10^6, 4
    PATH = "../../data/sampling_results/Benchmark-7/"
    
    for (ind, vals) in enumerate(data)
        
        print("Sampling event #$ind $(event_ind[ind]) out of $(length(event_ind)) \n")
        
        log_likelihood = log_lik_ndiff(vals, conv_mat)
        
        posterior = PosteriorDensity(log_likelihood, prior)
        
        sampler = MetropolisHastings(tuning=tuning,)
        
        algorithm = MCMCSampling(sampler=sampler, 
            nchains=nchains, 
            init=init, 
            burnin=burnin, 
            convergence=convergence
        )
        @time samples = bat_sample(
            posterior, nchains*nsamples, algorithm,
            max_neval = nchains*nsamples,
            max_time = Inf,
        ).result
        
        BAT.bat_write(PATH*"lc-$(event_ind[ind]).hdf5", unshaped.(samples))
    end
    
    
end

ind_samples =  [1, 10, 11, 12, 13, 137, 14, 15, 151, 16, 169, 17, 18, 19, 2, 20, 21, 22, 225, 23, 24, 25, 262, 299, 3, 343, 355, 369, 4, 48, 5, 6, 7, 8, 80, 9]

main(ind_samples)