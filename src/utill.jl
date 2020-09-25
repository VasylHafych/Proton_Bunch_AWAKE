
"""
    Generate envelope trajectories for multiple parameters. No resolution effects included. The first trajectory will be filled with a gray color and can be considered as a "truth" trajectory.  
"""
function plot_envelop_trajectory(
        params_array; 
        colors = ["gray", "red", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        labels = [i for i in 1:9]
        ) 
    
    fig, ax = plt.subplots(2,1, figsize=(8,6 ), sharex=true)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    
    x_range = range(-1, stop = 24, length=100)
    
    for (ind, params) in enumerate(params_array)
        
        σ_x(x) = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - x)^2) 
        σ_y(x) = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - x)^2) 

        σ_x_vals = [σ_x(x) for x in x_range]
        σ_y_vals = [σ_y(x) for x in x_range]

        ax[1].set_ylim(0, maximum(σ_x_vals))
        ax[2].set_ylim(0, maximum(σ_y_vals))
        
        if ind == 1 
            
#             ax[1].vlines(params.s_cam, 0,  maximum(σ_x_vals), linestyle="-", color="darkslategray", alpha=0.5)
#             ax[2].vlines(params.s_cam, 0,  maximum(σ_y_vals), linestyle="-", color="darkslategray", alpha=0.5)
            
            ax[1].axvline(params.s_cam[1], linestyle="-", color="darkslategray", alpha=0.5)
            ax[1].axvline(params.s_cam[2], linestyle="-", color="darkslategray", alpha=0.5)
            ax[1].axvline(params.s_cam[3], linestyle="-", color="darkslategray", alpha=0.5)
            ax[1].axvline(params.s_cam[4], linestyle="-", color="darkslategray", alpha=0.5)
            
            ax[2].axvline(params.s_cam[1], linestyle="-", color="darkslategray", alpha=0.5)
            ax[2].axvline(params.s_cam[2], linestyle="-", color="darkslategray", alpha=0.5)
            ax[2].axvline(params.s_cam[3], linestyle="-", color="darkslategray", alpha=0.5)
            ax[2].axvline(params.s_cam[4], linestyle="-", color="darkslategray", alpha=0.5)

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

    ax[1].legend(loc="upper left", ncol=5, framealpha=1)
    ax[1].set_ylabel(L"\sigma_x, [mm]")
    ax[2].set_ylabel(L"\sigma_y, [mm]")
    ax[2].set_xlabel(L"s, [m]")
end



"""
    Plot crosssection of signal and model prediction. The alignment plain is determined by the first parameter.  
"""
function plot_cam_crossections(params_array, data, conv_mat; 
        colors = ["C0", "C1", "C2"],
        labels=["1", "2", "3"],
        light_fluctuations = 2.0, 
        include_satur = false
        ) 
    
    fig, ax = plt.subplots(4,2, figsize=(8,7))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    [ax[i].set_xticks([]) for i in 1:8]
    [ax[i,2].set_yticks([]) for i in 1:4]
    
    
    for cam_ind in 1:4
        x_ind = round(Int64, params_array[1].algmx[cam_ind])
        y_ind = round(Int64, params_array[1].algmy[cam_ind])
        
        x_axis = 1:length(data[cam_ind][x_ind,:])
        y_axis = 1:length(data[cam_ind][:,y_ind])
        
#         ax[cam_ind, 1].plot(x_axis, data[cam_ind][x_ind,:], color="red", linewidth=1)
#         ax[cam_ind, 2].plot(y_axis, data[cam_ind][:,y_ind], color="red", linewidth=1)
        
        ax[cam_ind, 1].fill_between(x_axis, data[cam_ind][x_ind,:], color="gray", alpha=0.5, linewidth=0, label="Data")
        ax[cam_ind, 2].fill_between(y_axis, data[cam_ind][:,y_ind], color="gray", alpha=0.5, linewidth=0)
        
#         ax[cam_ind, 1].set_yscale("log")
#         ax[cam_ind, 2].set_yscale("log")
    end
    
    for (ind, params) in enumerate(params_array)
        simulated_data = generate_event(params, data.population, conv_mat; inc_noise=false, size=[size(data.cam_1), size(data.cam_2), size(data.cam_3), size(data.cam_4)], include_satur=include_satur)
        
        for cam_ind in 1:4
            x_ind = round(Int64, params_array[1].algmx[cam_ind])
            y_ind = round(Int64, params_array[1].algmy[cam_ind])
            
            x_axis = 1:length(simulated_data[cam_ind][x_ind,:])
            y_axis = 1:length(simulated_data[cam_ind][:,y_ind])
            
            if ind == 1 
                ax[cam_ind, 1].plot(x_axis, simulated_data[cam_ind][x_ind,:], color=colors[ind], alpha=1, linewidth=1.5, label=labels[ind])
                ax[cam_ind, 2].plot(y_axis, simulated_data[cam_ind][:,y_ind], color=colors[ind], alpha=1, linewidth=1.5)
            else
                ax[cam_ind, 1].plot(x_axis, simulated_data[cam_ind][x_ind,:], label=labels[ind], color=colors[ind])
                ax[cam_ind, 2].plot(y_axis, simulated_data[cam_ind][:,y_ind], color=colors[ind])
                
            end
        end
        
    end
    
    ax[1].legend(loc="upper left")
    ax[4,1].set_xlabel(L"x")
    ax[4,2].set_xlabel(L"y")
    
    fig.text(0.04, 0.5, "Pixel Value", va="center", rotation="vertical")

end

"""
    Plot integral of signal and model prediction.  
"""
function plot_cam_integral(params_array, data, conv_mat; 
        colors = ["C0", "C1", "C2"],
        labels=["1", "2", "3"],
        light_fluctuations = 2.0,
        include_satur = false
        ) 
    
    fig, ax = plt.subplots(4,2, figsize=(8,7))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    [ax[i].set_xticks([]) for i in 1:8]
    [ax[i,2].set_yticks([]) for i in 1:4]
    
    
    for cam_ind in 1:4
        
        x_axis = 1:length(data[cam_ind][1,:])
        y_axis = 1:length(data[cam_ind][:,1])
              
        ax[cam_ind, 1].fill_between(x_axis, [sum(data[cam_ind], dims=1)...], color="gray", alpha=0.5, linewidth=0, label="Data")
        ax[cam_ind, 2].fill_between(y_axis, [sum(data[cam_ind], dims=2)...], color="gray", alpha=0.5, linewidth=0)
    
    end
    
    for (ind, params) in enumerate(params_array)
        
        simulated_data = generate_event(params, data.population, conv_mat; inc_noise=false, size=[size(data.cam_1), size(data.cam_2), size(data.cam_3), size(data.cam_4)], include_satur=include_satur)
        
        for cam_ind in 1:4
            x_ind = round(Int64, params.algmx[cam_ind])
            y_ind = round(Int64, params.algmy[cam_ind])
            
            x_axis = 1:length(simulated_data[cam_ind][x_ind,:])
            y_axis = 1:length(simulated_data[cam_ind][:,y_ind])
            
            if ind == 1 
                ax[cam_ind, 1].plot(x_axis, [sum(simulated_data[cam_ind], dims=1)...], color=colors[ind], alpha=1, linewidth=1.5, label=labels[ind])
                ax[cam_ind, 2].plot(y_axis, [sum(simulated_data[cam_ind], dims=2)...], color=colors[ind], alpha=1, linewidth=1.5)
            else
                ax[cam_ind, 1].plot(x_axis, [sum(simulated_data[cam_ind], dims=1)...], label=labels[ind], color=colors[ind])
                ax[cam_ind, 2].plot(y_axis, [sum(simulated_data[cam_ind], dims=2)...], color=colors[ind])
                
            end
        end
        
    end
    
    ax[1].legend(loc="upper left")
    ax[4,1].set_xlabel(L"x")
    ax[4,2].set_xlabel(L"y")
    
    fig.text(0.04, 0.5, "Pixel Value", va="center", rotation="vertical")

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