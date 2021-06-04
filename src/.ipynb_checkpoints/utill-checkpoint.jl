"""
    Plot crosssection of signal and model prediction. The alignment plain is determined by the first parameter.  
"""
function plot_cam_crossections(params_array, data, conv_mat; 
        colors = ["C0", "C1", "C2"],
        labels=["1", "2", "3"],
        light_fluctuations = 2.0, 
        include_satur = false,
        figsize=(8,7)
        ) 
    
    fig, ax = plt.subplots(4,2, figsize=figsize)
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
        include_satur = false,
        figsize=(8,7)
        ) 
    
    fig, ax = plt.subplots(4,2, figsize=figsize)
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
        figsize = figsize,
        saveplot = false,
        filename = false
        
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
    
    if saveplot
        fig.savefig(filename, bbox_inches = "tight") 
    end
        
end

function eval_conv(conv_matrix, ind)
    ind = convert.(Int64, ind)
    y_tmp = exp.(conv_matrix[:, ind[1]+1])
    for ind_tmp in ind[2:end]
        y_tmp = conv(y_tmp, exp.(conv_matrix[:, ind_tmp+1]))
    end
    prepend!(y_tmp, repeat([0], length(ind)-1))
end

function eval_quantile(conv_matrix, ind; alpha_min = 0.025, alpha_max = 0.975)
    vals = eval_conv(conv_matrix, ind)
    vals_up = argmin(abs.(cumsum(vals) .- alpha_max)) 
    vals_down = argmin(abs.(cumsum(vals) .- alpha_min))
    return (vals_down, vals_up)
end

function eval_conv_is(ind, lf)
    mue = sum(ind)
    sigma = sqrt(sum((lf .* sqrt.(ind)).^2))
    return truncated(Normal(mue, sigma), 0, Inf)
end

nansum(x) = sum(x[.!isnan.(x)])

function plot_projections(
        cv_matrix, 
        event_tr,
        event_nt,
        params; 
        isnontr = false, 
        istrunc = true, 
        figsize=(12,8),
        saveplot = false,
        filename = false
    )
    
    amp_coeff = 1.15
    alpha_1 = 0.005
    alpha_2 = 0.995
    
    color_1 = "darkgray"
    color_2 = "k"
    color_3 = "red"
    
    median_event = generate_event(params, 
        event_nt.population, cv_matrix; 
        inc_noise=false, 
        size=[size(event_nt.cam_1), size(event_nt.cam_2), size(event_nt.cam_3), size(event_nt.cam_4)], 
        include_satur=false
    )
    
    fig, ax = plt.subplots(4,2, figsize=figsize)
    fig.subplots_adjust(hspace=0.24, wspace=0.05)
    
    for i in 1:4 
        ycounts_nt = [sum(event_nt[i], dims=1)...]
        ycounts_tr = [sum(event_tr[i], dims=1)...]
        median_sum = [sum(median_event[i], dims=1)...]

        xedges = 1:length(ycounts_tr)
        
        if isnontr
            ax[i,1].step(xedges, ycounts_nt, color=color_1, where="mid", zorder=0)
        end
        
        if istrunc
            ax[i,1].fill_between(xedges, ycounts_tr, step="mid", color=color_1, alpha=0.8,label="Data")
        end

        if i != 4
            fluct = [eval_quantile(conv_matrices[i], j, alpha_min = alpha_1, alpha_max = alpha_2) for j in eachcol(median_event[i])]
            fluct_up = [j[1] for j in fluct] .- median_sum
            fluct_down = median_sum .- [j[2] for j in fluct] 
        else
            fluct = [eval_conv_is(j, params.cam4_light_fluct) for j in eachcol(median_event[i])];
            fluct_up = [quantile(j, 0.975) for j in fluct] .- median_sum
            fluct_down = median_sum .- [quantile(j, 0.025) for j in fluct]
        end

        ax[i,1].errorbar(xedges, median_sum, yerr=[fluct_down, fluct_up], ms=2.2, fmt=".", color=color_2, ecolor=color_3,  capthick=0.5, capsize=1.2, linewidth=0.5, label="Model")

        ax[i,1].set_ylim(bottom=0.0)
        ax[i,1].set_xlim(minimum(xedges), maximum(xedges))

        ycounts_nt = [sum(event_nt[i], dims=2)...]
        ycounts_tr = [sum(event_tr[i], dims=2)...]
        median_sum = [sum(median_event[i], dims=2)...]

        xedges = 1:length(ycounts_tr)

        if isnontr
            ax[i,2].step(xedges, ycounts_nt, color=color_1, where="mid", zorder=0)
        end
        if istrunc
            ax[i,2].fill_between(xedges, ycounts_tr, step="mid", color=color_1, alpha=0.8, label="Data")
        end

        if i != 4
            fluct = [eval_quantile(conv_matrices[i], j, alpha_min = alpha_1, alpha_max = alpha_2) for j in eachrow(median_event[i])]
            fluct_up = [j[1] for j in fluct] .- median_sum
            fluct_down = median_sum .- [j[2] for j in fluct] 
        else
            fluct = [eval_conv_is(j, params.cam4_light_fluct) for j in eachrow(median_event[i])];
            fluct_up = [quantile(j, 0.975) for j in fluct] .- median_sum
            fluct_down = median_sum .- [quantile(j, 0.025) for j in fluct];
        end

        ax[i,2].errorbar(xedges, median_sum, yerr=[fluct_down, fluct_up], ms=2.2, fmt=".", color=color_2, ecolor=color_3,  capthick=0.5, capsize=1.2, linewidth=0.5, label="Model")

        ax[i,2].set_ylim(bottom=0.0)
        ax[i,2].set_xlim(minimum(xedges), maximum(xedges))

        ax[i,1].set_yticks([])
        ax[i,2].set_yticks([])
    end

    ax[4,1].set_xlabel("y (pixels)")
    ax[4,2].set_xlabel("x (pixels)")
    
    ax[1,2].legend(loc="upper right", framealpha=0.0)

    ax[1,1].set_ylabel("Cam. 1")
    ax[2,1].set_ylabel("Cam. 2")
    ax[3,1].set_ylabel("Cam. 3")
    ax[4,1].set_ylabel("Cam. 4")

    fig.text(0.05, 0.5, "Integrated Signal (a.u.)", va="center", rotation="vertical")
    
    if saveplot
        fig.savefig(filename, bbox_inches = "tight") 
    end
end

function plot_projections_prl(
        cv_matrix, 
        event_tr,
        event_nt,
        params; 
        isnontr = false, 
        istrunc = true, 
        figsize=(12,8),
        saveplot = false,
        filename = false
    )
    
    amp_coeff = 1.15
    alpha_1 = 0.005
    alpha_2 = 0.995
    
    color_1 = "darkgray"
    color_2 = "k"
    color_3 = "red"
    
    median_event = generate_event(params, 
        event_nt.population, cv_matrix; 
        inc_noise=false, 
        size=[size(event_nt.cam_1), size(event_nt.cam_2), size(event_nt.cam_3), size(event_nt.cam_4)], 
        include_satur=false
    )
    
    fig, ax = plt.subplots(2,4, figsize=figsize)
    fig.subplots_adjust(hspace=0.0, wspace=0.05)
    
    for i in 1:4 
        ycounts_nt = [sum(event_nt[i], dims=1)...]
        ycounts_tr = [sum(event_tr[i], dims=1)...]
        median_sum = [sum(median_event[i], dims=1)...]

        xedges = 1:length(ycounts_tr)
        
        if isnontr
            ax[1, i].step(xedges, ycounts_nt, color="darkblue", where="mid", lw=0.8, zorder=1, label="Data")
        end
        
#         if istrunc
#             ax[1, i].fill_between(xedges, ycounts_tr, color=color_1, alpha=0.8, lw=0.0, label="Data")
#         end

        if i != 4
            fluct = [eval_quantile(conv_matrices[i], j, alpha_min = alpha_1, alpha_max = alpha_2) for j in eachcol(median_event[i])]
            fluct_up = [j[1] for j in fluct] .- median_sum
            fluct_down = median_sum .- [j[2] for j in fluct] 
        else
            fluct = [eval_conv_is(j, params.cam4_light_fluct) for j in eachcol(median_event[i])];
            fluct_up = [quantile(j, 0.975) for j in fluct] .- median_sum
            fluct_down = median_sum .- [quantile(j, 0.025) for j in fluct]
        end
        
        ax[1,i].fill_between(xedges, median_sum .- fluct_down, median_sum .+ fluct_up, color=:gray,  lw=0.0, alpha=0.5, label=L"90\%")
        ax[1,i].scatter(xedges, median_sum, color="darkred", alpha=1.0, s=1, label="Model")
#         ax[1,i].errorbar(xedges, median_sum, yerr=[fluct_down, fluct_up], ms=2.2, fmt=".", color=color_2, ecolor=color_3,  errorevery=2, capthick=0.5, capsize=1.2, linewidth=0.5, label="Model")

        ax[1, i].set_ylim(bottom=0.0)
        ax[1, i].set_xlim(minimum(xedges), maximum(xedges))

        ycounts_nt = [sum(event_nt[i], dims=2)...]
        ycounts_tr = [sum(event_tr[i], dims=2)...]
        median_sum = [sum(median_event[i], dims=2)...]

        xedges = 1:length(ycounts_tr)

        if isnontr
            ax[2, i].step(xedges, ycounts_nt, color="darkblue", where="mid", lw=0.8, zorder=1)
        end


        if i != 4
            fluct = [eval_quantile(conv_matrices[i], j, alpha_min = alpha_1, alpha_max = alpha_2) for j in eachrow(median_event[i])]
            fluct_up = [j[1] for j in fluct] .- median_sum
            fluct_down = median_sum .- [j[2] for j in fluct] 
        else
            fluct = [eval_conv_is(j, params.cam4_light_fluct) for j in eachrow(median_event[i])];
            fluct_up = [quantile(j, 0.975) for j in fluct] .- median_sum
            fluct_down = median_sum .- [quantile(j, 0.025) for j in fluct];
        end
    
        ax[2,i].fill_between(xedges, median_sum .- fluct_down, median_sum .+ fluct_up, color=:gray, lw=0.0, alpha=0.5)
        ax[2,i].scatter(xedges, median_sum, color="darkred", alpha=1.0, s=1,)
        
        ax[2, i].set_ylim(bottom=0.0)
        ax[2, i].set_xlim(minimum(xedges), maximum(xedges))

        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[2, i].set_yticks([])
    end

    ax[1,1].set_ylabel("Projection Y")
    ax[2,1].set_ylabel("Projection X")
    
    ax[1,4].legend(loc="upper right", framealpha=0.0, bbox_to_anchor=(1.6, 1.0))

    ax[1,1].set_title("Cam. 1")
    ax[1,2].set_title("Cam. 2")
    ax[1,3].set_title("Cam. 3")
    ax[1,4].set_title("Cam. 4")

    fig.text(0.07, 0.45, "Integrated Signal (a.u.)", va="center", rotation="vertical")
    fig.text(0.45, 0.02, "Pixel Number", va="center", rotation="horizontal")
    
    if saveplot
        fig.savefig(filename, bbox_inches = "tight") 
    end
end
