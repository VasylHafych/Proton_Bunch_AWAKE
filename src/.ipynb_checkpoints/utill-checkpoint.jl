function plot_envelop_trajectory(
        params_array; 
        colors = ["gray", "red", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        labels = [i for i in 1:9]
        ) 
    
    fig, ax = plt.subplots(2,1, figsize=(6,4), sharex=true)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    
    x_range = range(-1, stop = 24, length=100)
    
    for (ind, params) in enumerate(params_array)
        
        σ_x(x) = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.s_waist[1] - x)^2) 
        σ_y(x) = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.s_waist[1] - x)^2) 

        σ_x_vals = [σ_x(x) for x in x_range]
        σ_y_vals = [σ_y(x) for x in x_range]

        ax[1].set_ylim(0, maximum(σ_x_vals))
        ax[2].set_ylim(0, maximum(σ_y_vals))
        
        if ind == 1 
            
            ax[1].vlines(params.s_cam, 0,  maximum(σ_x_vals), linestyle="-", color="darkslategray", alpha=0.5)
            ax[2].vlines(params.s_cam, 0,  maximum(σ_y_vals), linestyle="-", color="darkslategray", alpha=0.5)

            ax[1].fill_between(x_range, σ_x_vals, color=colors[ind], label=labels[ind], alpha=0.4)
            ax[2].fill_between(x_range, σ_y_vals, color=colors[ind], label=labels[ind],  alpha=0.4)

            ax[1].set_xlim(-1, maximum(x_range))

            ax[1].set_ylim(0, maximum(σ_x_vals))
            ax[2].set_ylim(0, maximum(σ_y_vals))
        else 
            ax[1].plot(x_range, σ_x_vals, color=colors[ind], ls="--", label=labels[ind])
            ax[2].plot(x_range, σ_y_vals, color=colors[ind], ls="--", label=labels[ind])
        end
    end

    ax[1].legend(loc="upper left", ncol=5, framealpha=0.0)
    ax[1].set_ylabel(L"\sigma_x")
    ax[2].set_ylabel(L"\sigma_y")
    ax[2].set_xlabel(L"s")
end

function generate_event(
        params::D, population::Float64, conv_mat::T; 
        inc_noise=true,
        size = [(81, 101),(101, 101),(41, 51),(241, 351)],
        light_fluctuations = 2.0
    ) where {T<: NamedTuple, D <: NamedTuple}

    
    img_1 = generate_image(params, population, conv_mat.cam_1, light_fluctuations, 1, size = size[1], inc_noise=inc_noise)
    img_2 = generate_image(params, population, conv_mat.cam_2, light_fluctuations, 2, size = size[2], inc_noise=inc_noise)
    img_3 = generate_image(params, population, conv_mat.cam_3, light_fluctuations, 3, size = size[3], inc_noise=inc_noise)
    img_4 = generate_image(params, population, conv_mat.cam_4, light_fluctuations, 4, size = size[4], inc_noise=inc_noise)
    
    return (cam_1 = img_1, cam_2 = img_2, cam_3 = img_3, cam_4 = img_4, population = population)
end

function total_likelihood(params::D, data::M, conv_mat::T) where {T<: NamedTuple, D <: NamedTuple, M <: NamedTuple}
    
    ll = zero(Float64)
        
    ll += cam_likelihood(params, data.cam_1, data.population, conv_mat.cam_1, 1)
    ll += cam_likelihood(params, data.cam_2, data.population, conv_mat.cam_2, 2)
    ll += cam_likelihood(params, data.cam_3, data.population, conv_mat.cam_3, 3)
    ll += cam_likelihood(params, data.cam_4, data.population, conv_mat.cam_4, 4)
    
    return ll
end

function plot_cam_crossections(params_array, data, conv_mat; 
        colors = ["gray", "orange"], 
        light_fluctuations = 2.0
        ) 
    
    fig, ax = plt.subplots(4,2, figsize=(6,6))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    [ax[i].set_xticks([]) for i in 1:8]
    [ax[i,2].set_yticks([]) for i in 1:4]
    
    
    for cam_ind in 1:4
        x_ind = round(Int64, params_array[1].μ_x[cam_ind])
        y_ind = round(Int64, params_array[1].μ_y[cam_ind])
        
        x_axis = 1:length(data[cam_ind][x_ind,:])
        y_axis = 1:length(data[cam_ind][:,y_ind])
        
        ax[cam_ind, 1].plot(x_axis, data[cam_ind][x_ind,:], color="red", linewidth=1)
        ax[cam_ind, 2].plot(y_axis, data[cam_ind][:,y_ind], color="red", linewidth=1)
        
        ax[cam_ind, 1].set_yscale("log")
        ax[cam_ind, 2].set_yscale("log")
    end
    
    for (ind, params) in enumerate(params_array)
        simulated_data = generate_event(params, data.population, conv_mat; inc_noise=false, size=[size(data.cam_1), size(data.cam_2), size(data.cam_3), size(data.cam_4)])
        
        for cam_ind in 1:4
            x_ind = round(Int64, params.μ_x[cam_ind])
            y_ind = round(Int64, params.μ_y[cam_ind])
            
            x_axis = 1:length(simulated_data[cam_ind][x_ind,:])
            y_axis = 1:length(simulated_data[cam_ind][:,y_ind])
            
            if ind == 1 
                ax[cam_ind, 1].fill_between(x_axis, simulated_data[cam_ind][x_ind,:], color=colors[ind], alpha=0.5)
                ax[cam_ind, 2].fill_between(y_axis, simulated_data[cam_ind][:,y_ind], color=colors[ind], alpha=0.5)
            else
                ax[cam_ind, 1].plot(x_axis, simulated_data[cam_ind][x_ind,:])
                ax[cam_ind, 2].plot(y_axis, simulated_data[cam_ind][:,y_ind])
                
            end
        end
        
    end

end

function corner_plots(
        samples, 
        dim_indices::AbstractArray, 
        dim_names::AbstractArray;
        N_bins = 50,
        levels_quantiles = [0.4, 0.7, 0.8, 0.9, 0.99, 1,], 
        hist_color = plt.cm.Blues(0.7), 
        colors = vcat([1 1 1 1], plt.cm.Blues(range(0, stop=1, length=length(levels_quantiles)))[2:end,:]),
        figsize = figsize
    )
    
    sample_weights = samples.weight
    samples = flatview(unshaped.(samples.v))
    
    N = length(dim_indices)
    bins=[] #Vector{StepRangeLen}()
    fig, ax = plt.subplots(N,N, figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for idx in 1:N
        dim_idx = dim_indices[idx]
        bins_tmp = range(minimum(samples[dim_idx,:]), stop=maximum(samples[dim_idx,:]), length=N_bins)
        push!(bins, bins_tmp)
        ax[idx, idx].hist(samples[dim_idx,:], weights=sample_weights, bins=bins_tmp, color=hist_color)
        ax[idx, idx].set_xlim(first(bins_tmp),last(bins_tmp))
    end
    
    for i in 2:N, j in 1:(i-1)
        dim_x = dim_indices[j]
        dim_y = dim_indices[i]
        histogram_2D = fit(Histogram, (samples[dim_x,:],samples[dim_y,:]), weights(sample_weights), (bins[j], bins[i]))
        histogram_2D = normalize(histogram_2D, mode=:probability)
        
        levels=quantile([histogram_2D.weights...], levels_quantiles)
        
        ax[i,j].contourf(midpoints(histogram_2D.edges[1]), midpoints(histogram_2D.edges[2]), histogram_2D.weights', levels=levels, colors=colors)
        ax[i,j].set_xlim(first(bins[j]),last(bins[j]))
        ax[i,j].set_ylim(first(bins[i]),last(bins[i]))
        ax[j,i].set_visible(false)
        
    end
    
    for i in 1:N, j in 1:N
        if i < N 
            ax[i,j].get_xaxis().set_visible(false)
        else
            ax[i,j].set_xlabel(dim_names[j])
        end
        
        if j == i || j>1
           ax[i,j].get_yaxis().set_visible(false) 
        else
            ax[i,j].set_ylabel(dim_names[i])
        end
    end
        
end