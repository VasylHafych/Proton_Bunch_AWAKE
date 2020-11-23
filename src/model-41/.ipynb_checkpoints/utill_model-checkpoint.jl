"""
    Generate envelope trajectories for multiple parameters. No resolution effects included. The first trajectory will be filled with a gray color and can be considered as a "truth" trajectory.  
"""
function plot_envelop_trajectory(
        params_array; 
        colors = ["gray", "red", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        labels = [i for i in 1:9],
        figsize=(8,6)
        ) 
    
    fig, ax = plt.subplots(2,1, figsize=figsize, sharex=true)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    
    x_range = range(-1, stop = 24, length=100)
    
    for (ind, params) in enumerate(params_array)
        
        σ_x(x) = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - x)^2) 
        σ_x_2(x) = sqrt.(params.tr_size_2[1]^2 + 10^-4*params.ang_spr_2[1]^2*(params.waist_2[1] - x)^2) 
        
        σ_y(x) = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - x)^2) 
        σ_y_2(x) = sqrt.(params.tr_size_2[2]^2 + 10^-4*params.ang_spr_2[2]^2*(params.waist_2[1] - x)^2) 

        σ_x_vals = [σ_x(x) for x in x_range]
        σ_y_vals = [σ_y(x) for x in x_range]
        
        σ_x_vals_2 = [σ_x_2(x) for x in x_range]
        σ_y_vals_2 = [σ_y_2(x) for x in x_range]

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

            ax[1].plot(x_range, σ_x_vals, color="k", ls="--", label="Halo")
            ax[2].plot(x_range, σ_y_vals, color="k", ls="--",)
            
            ax[1].plot(x_range, σ_x_vals_2, color="k", ls=":", label="Core")
            ax[2].plot(x_range, σ_y_vals_2, color="k", ls=":",)
            
            ax[1].set_xlim(-1, maximum(x_range))

            ax[1].set_ylim(0, maximum(σ_x_vals))
            ax[2].set_ylim(0, maximum(σ_y_vals))
        else 
            ax[1].plot(x_range, σ_x_vals, color="k", ls="--", )
            ax[2].plot(x_range, σ_y_vals, color="k", ls=":",)
            
            ax[1].plot(x_range, σ_x_vals_2, color="k", ls="--", )
            ax[2].plot(x_range, σ_y_vals_2, color="k", ls=":",)
        end
    end

    ax[1].legend(loc="upper left", ncol=5, framealpha=1)
    ax[1].set_ylabel(L"\sigma_x, [mm]")
    ax[2].set_ylabel(L"\sigma_y, [mm]")
    ax[2].set_xlabel(L"s, [m]")
end
