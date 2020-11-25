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

function plot_signal_envelope(params)
    
    color_1 = "sandybrown"
    color_2 = "lightsteelblue"
    
    size = (10, 8)
    
    fig = plt.figure(figsize=size,)

    fig.subplots_adjust(hspace=0.00, wspace=0.00)

    r11 = plt.subplot2grid((13, 4), (0, 0), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    r12 = plt.subplot2grid((13, 4), (0, 1), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    r13 = plt.subplot2grid((13, 4), (0, 2), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    r14 = plt.subplot2grid((13, 4), (0, 3), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    rst1 = [r11, r12, r13, r14]

    r2 = plt.subplot2grid((13, 4), (3, 0), rowspan=3, colspan=4, )
    r2.tick_params(axis="x", pad=8)
    r3 = plt.subplot2grid((13, 4), (7, 0), rowspan=3, colspan=4, xticklabels=[], )

    r3.xaxis.tick_top()

    r41 = plt.subplot2grid((13, 4), (10, 0), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    r42 = plt.subplot2grid((13, 4), (10, 1), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    r43 = plt.subplot2grid((13, 4), (10, 2), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    r44 = plt.subplot2grid((13, 4), (10, 3), rowspan=3, colspan=1, yticks=[],  xticks=[], )
    rst2 = [r41, r42, r43, r44]
    
    # make plot
    
    x_range = range(-1, stop = 24, length=100)
    σ_x(x) = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - x)^2) 
    σ_x_2(x) = sqrt.(params.tr_size_2[1]^2 + 10^-4*params.ang_spr_2[1]^2*(params.waist_2[1] - x)^2) 

    σ_y(x) = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - x)^2) 
    σ_y_2(x) = sqrt.(params.tr_size_2[2]^2 + 10^-4*params.ang_spr_2[2]^2*(params.waist_2[1] - x)^2) 

    σ_x_vals = [σ_x(x) for x in x_range]
    σ_y_vals = [σ_y(x) for x in x_range]

    σ_x_vals_2 = [σ_x_2(x) for x in x_range]
    σ_y_vals_2 = [σ_y_2(x) for x in x_range]
    
    r2.fill_between(x_range, σ_x_vals, color=color_1, alpha=0.5, )
    r3.fill_between(x_range, σ_y_vals, color=color_1, alpha=0.5, label="Halo Component")
            
    r2.fill_between(x_range, σ_x_vals_2, color=color_2, alpha=0.5, )
    r3.fill_between(x_range, σ_y_vals_2, color=color_2, alpha=0.5, label="Core Component")
    r3.legend(bbox_to_anchor=(0.2, 0.01, 0.4, 0.15), ncol=2, framealpha=0.0)
    
    r2.axvline(params.s_cam[1], linestyle=":", color="darkslategray", alpha=0.5, label="Cam. Position")
    r2.axvline(params.s_cam[2], linestyle=":", color="darkslategray", alpha=0.5)
    r2.axvline(params.s_cam[3], linestyle=":", color="darkslategray", alpha=0.5)
    r2.axvline(params.s_cam[4], linestyle=":", color="darkslategray", alpha=0.5)
            
    r3.axvline(params.s_cam[1], linestyle=":", color="darkslategray", alpha=0.5)
    r3.axvline(params.s_cam[2], linestyle=":", color="darkslategray", alpha=0.5)
    r3.axvline(params.s_cam[3], linestyle=":", color="darkslategray", alpha=0.5)
    r3.axvline(params.s_cam[4], linestyle=":", color="darkslategray", alpha=0.5)
    
    r2.axvline(params.waist[1], linestyle="-", color=color_1, alpha=1, label="Waist Halo")
    r3.axvline(params.waist[1], linestyle="-", color=color_1, alpha=1)
    r2.axvline(params.waist_2[1], linestyle="-", color=color_2, alpha=1, label="Waist Core")
    r3.axvline(params.waist_2[1], linestyle="-", color=color_2, alpha=1)
    
    r2.axvline(2.9, linestyle=":", color="red", alpha=1, label="Waist Expected")
    r3.axvline(2.9, linestyle=":", color="red", alpha=1, label="Expected")
    
    r2.legend(bbox_to_anchor=(0.6, 0.98,), ncol=2, framealpha=0.0)
    
    for i in 1:4
        if i < 4
            npix = i < 3 ? 71 : 41
            x_bins = range(0, length = npix+1, step = params.psx[i]*10^-3)
            y_bins = range(0, length = npix+1, step = params.psy[i]*10^-3)
                
            mue_x = maximum(x_bins) / 2
            mue_y = maximum(y_bins) / 2
            
            σ_x_vls = σ_x(params.s_cam[i])
            σ_x_2_vls = σ_x_2(params.s_cam[i])
            
            σ_y_vls = σ_y(params.s_cam[i])
            σ_y_2_vls = σ_y_2(params.s_cam[i])
    
            vals_x_1 = diff(cdf.(Normal(mue_x, σ_x_vls), x_bins))
            vals_x_2 = diff(cdf.(Normal(mue_x, σ_x_2_vls), x_bins))

            vals_y_1 = diff(cdf.(Normal(mue_y, σ_y_vls), y_bins))
            vals_y_2 = diff(cdf.(Normal(mue_y, σ_y_2_vls), y_bins))
            
            rst1[i].fill_between(x_bins[1:end-1], params.mixt_pow .* vals_x_1, color=color_1, alpha=0.5, label="Halo")
            rst1[i].fill_between(x_bins[1:end-1], (1-params.mixt_pow).*vals_x_2, color=color_2, alpha=0.5, label="Core")
            rst1[i].plot(x_bins[1:end-1], (1-params.mixt_pow).*vals_x_2 .+ params.mixt_pow .* vals_x_1, color="k", ls="--", label="Sum")
            
            rst2[i].fill_between(y_bins[1:end-1], params.mixt_pow .* vals_y_1, alpha=0.5, color=color_1, )
            rst2[i].fill_between(y_bins[1:end-1], (1-params.mixt_pow).*vals_y_2, alpha=0.5, color=color_2, )
            rst2[i].plot(y_bins[1:end-1], (1-params.mixt_pow).*vals_y_2 .+ params.mixt_pow .* vals_y_1, color="k", ls="--")
            
        else
            npix = 71
            x_bins = range(0, length = npix+1, step = params.cam4_psx*10^-3)
            y_bins = range(0, length = npix+1, step = params.cam4_psy*10^-3)
            
            mue_x = maximum(x_bins) / 2
            mue_y = maximum(y_bins) / 2
            
            σ_x_vls = σ_x(params.s_cam[i])
            σ_x_2_vls = σ_x_2(params.s_cam[i])
            
            σ_y_vls = σ_y(params.s_cam[i])
            σ_y_2_vls = σ_y_2(params.s_cam[i])
    
            vals_x_1 = diff(cdf.(Normal(mue_x, σ_x_vls), x_bins))
            vals_x_2 = diff(cdf.(Normal(mue_x, σ_x_2_vls), x_bins))

            vals_y_1 = diff(cdf.(Normal(mue_y, σ_y_vls), y_bins))
            vals_y_2 = diff(cdf.(Normal(mue_y, σ_y_2_vls), y_bins))
            
            rst1[i].fill_between(x_bins[1:end-1], params.mixt_pow .* vals_x_1, color=color_1, alpha=0.5, label="Halo")
            rst1[i].fill_between(x_bins[1:end-1], (1-params.mixt_pow).*vals_x_2, color=color_2, alpha=0.5, label="Core")
            rst1[i].plot(x_bins[1:end-1], (1-params.mixt_pow).*vals_x_2 .+ params.mixt_pow .* vals_x_1, color="k",  ls="--", label="Sum")
            
            rst2[i].fill_between(y_bins[1:end-1], params.mixt_pow .* vals_y_1, color=color_1, alpha=0.5, )
            rst2[i].fill_between(y_bins[1:end-1], (1-params.mixt_pow).*vals_y_2, color=color_2, alpha=0.5, )
            rst2[i].plot(y_bins[1:end-1], (1-params.mixt_pow).*vals_y_2 .+ params.mixt_pow .* vals_y_1, color="k", ls="--")
            
        end
    end
    
    r2.text(0.015,0.62,"Cam. 1", rotation="vertical", color="darkslategray", transform=r2.transAxes) 
    r2.text(0.075,0.62,"Cam. 2", rotation="vertical", color="darkslategray", transform=r2.transAxes) 
    r2.text(0.62,0.62,"Cam. 3", rotation="vertical", color="darkslategray", transform=r2.transAxes) 
    r2.text(0.94,0.62,"Cam. 4", rotation="vertical", color="darkslategray", transform=r2.transAxes) 
    
    r11.text(0.04,0.89,"Cam. 1", color="darkslategray", transform=r11.transAxes)
    r12.text(0.04,0.89,"Cam. 2", color="darkslategray", transform=r12.transAxes)
    r13.text(0.04,0.89,"Cam. 3", color="darkslategray", transform=r13.transAxes)
    r14.text(0.04,0.89,"Cam. 4", color="darkslategray", transform=r14.transAxes)

    fig.text(0.92, 0.25, "Projection Y", rotation="vertical")
    fig.text(0.92, 0.65, "Projection X", rotation="vertical")
    
    r2.set_ylabel(L"\sigma_x \;(mm)")
    r3.set_ylabel(L"\sigma_y \;(mm)")
    
    r2.set_xlim(-1, maximum(x_range))
    r3.set_xlim(-1, maximum(x_range))
    
    max_sigma = maximum([maximum(σ_x_vals), maximum(σ_y_vals), maximum(σ_x_vals_2), maximum(σ_y_vals_2)])
    
    r2.set_ylim(bottom=0.0, top = max_sigma)
    r3.set_ylim(bottom=0.0, top = max_sigma)
    r3.invert_yaxis()

    r11.set_ylim(bottom=0.0)
    r12.set_ylim(bottom=0.0)
    r13.set_ylim(bottom=0.0)
    r14.set_ylim(bottom=0.0)

    r41.set_ylim(bottom=0.0)
    r42.set_ylim(bottom=0.0)
    r43.set_ylim(bottom=0.0)
    r44.set_ylim(bottom=0.0)
     
end