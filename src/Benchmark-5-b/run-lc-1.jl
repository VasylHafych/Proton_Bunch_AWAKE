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
using Random123

include("../model-41/likelihood.jl")

function def_conv_mat()
    conv_mat = load("../../data/experiment/dataset_2/m1/conv-matrix-upd-2.jld2")
    return (cam_1 = conv_mat["cam_1"], cam_2 = conv_mat["cam_2"], cam_3 = conv_mat["cam_3"], cam_4 = 8.52060) 
end

function def_data_vector(ev_ind)
    images = load("../../data/experiment/dataset_2/m1/images-satur.jld2")
    
    return [( cam_1 = images["cam_1"][i ,:,:], cam_2 = images["cam_2"][i ,:,:], 
            cam_3 = images["cam_3"][i ,:,:], cam_4 = images["cam_4"][i ,:,:], 
            population = images["charge"][i ,:][1],) for i in ev_ind]
end

function def_rem_ind()
    images = load("../../data/experiment/dataset_2/m1/images-satur.jld2")
    rem_ind = eachindex(images["charge"])[1:2:end]
    return shuffle(rem_ind)
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
        threshold = 1.15,
        corrected = false
    )

    init = MCMCChainPoolInit(
        init_tries_per_chain = 50 .. 150,
        nsteps_init = 1500
    )

    burnin = MCMCMultiCycleBurnin(
        max_ncycles = 160,
        nsteps_per_cycle = 40000
    )
    
    mcmcalgo = MetropolisHastings(
        weighting = RepetitionWeighting(),
        tuning = tuning
    )
    
    rng = Philox4x()
    
    return mcmcalgo, convergence, init, burnin, rng
end

function def_prior()
    β1 = 0.015
    β2 = 0.0077
    β3 = 0.0058 

    return NamedTupleDist(
        tr_size = [truncated(Normal(0.2, 0.04), 0.03, 0.19), truncated(Normal(0.2, 0.04), 0.03, 0.19)],
        tr_size_2 = [truncated(Normal(0.2, 0.04), 0.03, 0.19), truncated(Normal(0.2, 0.04), 0.03, 0.19)],
        ang_spr = [truncated(Normal(4.0, 2.0), 1.0, 8.0), truncated(Normal(4.0, 2.0), 1.0, 8.0)],
        ang_spr_2 = [truncated(Normal(4.0, 2.0), 1.0, 4.0), truncated(Normal(4.0, 2.0), 1.0, 4.0)],
        mixt_pow =  0.35 .. 1.0, 
        waist = [truncated(Normal(2.774, 0.03), 2.5, 3.6)],
        waist_2 = [truncated(Normal(2.774, 0.03), 2.5, 3.6)],
        algmx = [23.0 .. 48, 23.0 .. 48.0, 10.0 .. 30.0, 23.0 .. 48.0],
        algmy = [23.0 .. 48, 23.0 .. 48.0, 10.0 .. 30.0, 23.0 .. 48.0],
        cam4_ped = 4.0 .. 40.0,
        cam4_light_fluct = 1.0 .. 3.0,
        cam4_light_amp = 1.6 .. 9.9, 
        resx = [1, 1, 1], # 23, 24, 25, 
        resy = [1, 1, 1], # 26,27, 28, 
        cam4_resx = truncated(Normal(3, 1.5), 0, Inf),
        cam4_resy = truncated(Normal(3, 1.5), 0, Inf), 
        psx = [27.1, 21.6, 114.0], # 31, 32, 33
        psy = [30.5, 23.4, 125.0], # 34, 35, 36
        cam4_psx = 121.8, # 37
        cam4_psy = 120.0, # 38
        light_amp  = [1.0 .. 13.0 , 1.0 .. 17.0, 1.0 .. 5.0], # 1.0 .. 5.0
        s_cam = [0.0, 1.47799, 15.025999, 23.1644],
);

end

function main(event_ind)
    
    prior = def_prior()
    data = def_data_vector(event_ind)
    conv_mat = def_conv_mat()
    mcmcalgo, convergence, init, burnin, rng = def_settings()
    nsamples, nchains = 6*10^5, 4
    PATH = "../../data/sampling_results/Benchmark-5-b/"
    
    for (ind, vals) in enumerate(data)
        
        print("Sampling event #$ind $(event_ind[ind]) out of $(length(event_ind)) \n")
        
        log_likelihood = log_lik_ndiff(vals, conv_mat)
        posterior = PosteriorDensity(log_likelihood, prior)
        
        try 
            @time samples = bat_sample(
                rng, posterior,
                MCMCSampling(
                    mcalg = mcmcalgo,
                    trafo = NoDensityTransform(),
                    nchains = nchains,
                    nsteps = nsamples,
                    init = init,
                    burnin = burnin,
                    convergence = convergence,
                    strict = false,
                )
            ).result

            BAT.bat_write(PATH*"lc-$(event_ind[ind]).hdf5", unshaped.(samples))
        catch
            @show "Error"
        end
    end
    
    
end

ind_samples =  def_rem_ind() 


main(ind_samples)