using JLD2
using FileIO
using Statistics
using StatsBase 
using Distributions
using LinearAlgebra
using HDF5
using IntervalSets
using Random, ArraysOfArrays
using ValueShapes
using Measurements
using BAT
using ForwardDiff
using BenchmarkTools

@load "../../data/sampling_results/Benchmark-5/mcmc-summary-data-600.jld" data_save
summary_data = data_save[1]
sampling_ind = 1:600
n_events = length(sampling_ind)

prior_ang = NamedTupleDist(
    θ = [10^-15 .. 10^-4 for i in 1:n_events],
    α = [0.0 .. 2*pi  for i in 1:n_events], #0 .. 2*pi
    x_alignm = [-200 .. 200 for i in 1:3],
    y_alignm = [-200 .. 200 for i in 1:3],
    σ_x = [0.001 .. 50., 0.001 .. 100., 0.001 .. 100.],
    σ_y = [0.001 .. 50., 0.001 .. 100., 0.001 .. 100.],
)

function log_lik(; data = summary_data, event_ind = sampling_ind)
    
    s_cam = Float64[1.478, 15.026, 23.1150]
    
    return params -> begin   
        
        ll = 0.0
    
        for (ind, val) in enumerate(event_ind)
            
            x_expected = cos(params.α[ind]).*params.θ[ind]*s_cam.*10^6
            y_expected = sin(params.α[ind]).*params.θ[ind]*s_cam.*10^6
            
            x_expected += data.μx_align[val][1] .+ params.x_alignm 
            y_expected += data.μy_align[val][1] .+ params.y_alignm
            
            ll += sum(logpdf.(Normal.(x_expected, params.σ_x), data.μx_align[val][2:end] ))
            ll += sum(logpdf.(Normal.(y_expected, params.σ_y), data.μy_align[val][2:end] ))
            
        end
        return LogDVal(ll)
    end
        
end

log_likelihood = log_lik()

posterior = PosteriorDensity(log_likelihood, prior_ang)

posterior_is = bat_transform(PriorToGaussian(), posterior, PriorSubstitution()).result;

iter = 100000
iter_warmup = 6000
chains = 4;

metric = BAT.DiagEuclideanMetric()
integrator = BAT.LeapfrogIntegrator()
proposal = BAT.NUTS(:MultinomialTS, :ClassicNoUTurn)
adaptor = BAT.StanHMCAdaptor(0.8, iter_warmup)
hmc_sampler = HamiltonianMC(metric, ForwardDiff, integrator, proposal, adaptor)

@time samples_is = bat_sample(posterior_is, iter, MCMCSampling(sampler = hmc_sampler, nchains = chains));

samples = samples_is.result

trafo_is = trafoof(posterior_is.likelihood)
samples = inv(trafo_is).(samples)

BAT.bat_write("../../data/sampling_results/Benchmark-9/samples-1-600.hdf5", unshaped.(samples))