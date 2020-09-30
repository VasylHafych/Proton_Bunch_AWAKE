"""
    Generate simulated event for the given parameters (camera 1-3). 
"""
function generate_image_cam13(
        params::T, 
        population::Float64,
        cv_matrix::Array{Float64,2},
        light_fluctuations::Float64,
        cam_ind::Int64;
        size::Tuple{Int64, Int64}=(101,101),
        inc_noise = true,
        include_satur = true
    ) where {T <: NamedTuple}
    
    image_matrix = zeros(Float64, size...)
    light_coefficient::Float64 = params.light_amp[cam_ind] * 10^5
    
    δ_x::Float64 = params.psx[cam_ind] * 10^-3
    δ_y::Float64 = params.psy[cam_ind] * 10^-3
    
    μ_x::Float64  = params.algmx[cam_ind] * δ_x
    μ_y::Float64  = params.algmy[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.resx[cam_ind]*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.resy[cam_ind]*δ_y).^2)
    
    bck_cumsum = exp.(cumsum(cv_matrix[:,1]))
    
    for pix_ind in CartesianIndices(image_matrix)
    
        x_edge::Float64 = pix_ind.I[1] * δ_x
        y_edge::Float64 = pix_ind.I[2] * δ_y

        pix_prediction::Float64 = cdf(Normal(μ_x,σ_x), x_edge) - cdf(Normal(μ_x,σ_x), x_edge - δ_x)
        pix_prediction *= cdf(Normal(μ_y,σ_y), y_edge) - cdf(Normal(μ_y,σ_y), y_edge - δ_y)
        
        pix_prediction = pix_prediction*light_coefficient
        
        if inc_noise
            pix_prediction = rand(truncated(Normal(pix_prediction, 0.5+light_fluctuations*sqrt(pix_prediction)), 0, 4096))
            background_tmp = bck_cumsum .- rand()
            background_tmp[background_tmp .< 0 ] .= Inf
            pix_prediction += argmin(background_tmp) - 1
        end
        
        if include_satur && pix_prediction > 4095
            pix_prediction = 4095
        end
        
        image_matrix[pix_ind] = round(Int64, pix_prediction)
    end

    return image_matrix
end

"""
    Generate simulated event for the given parameters (camera 4). 
"""
function generate_image_cam4(
        params::T, 
        population::Float64,
        cam_ind::Int64;
        size::Tuple{Int64, Int64}=(101,101),
        inc_noise = true,
        include_satur = true
    ) where {T <: NamedTuple}
    
    image_matrix = zeros(Float64, size...)
    light_coefficient::Float64 = params.cam4_light_amp * 10^5
    
    δ_x::Float64 = params.cam4_psx * 10^-3
    δ_y::Float64 = params.cam4_psy * 10^-3
    
    μ_x::Float64  = params.algmx[cam_ind] * δ_x
    μ_y::Float64  = params.algmy[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.cam4_resx*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.cam4_resy*δ_y).^2)
    
    for pix_ind in CartesianIndices(image_matrix)
    
        x_edge::Float64 = pix_ind.I[1] * δ_x
        y_edge::Float64 = pix_ind.I[2] * δ_y

        pix_prediction::Float64 = cdf(Normal(μ_x,σ_x), x_edge) - cdf(Normal(μ_x,σ_x), x_edge - δ_x)
        pix_prediction *= cdf(Normal(μ_y,σ_y), y_edge) - cdf(Normal(μ_y,σ_y), y_edge - δ_y)
        
        pix_prediction = pix_prediction*light_coefficient + params.cam4_ped
        
        if inc_noise
            pix_prediction = rand(truncated(Normal(pix_prediction, params.cam4_light_fluct*sqrt(pix_prediction)), 0.0, 4095)) 
#             pix_prediction = rand(truncated(Poisson(pix_prediction), 0.0, 4095)) 
        end
        
        if include_satur && pix_prediction > 4095
            pix_prediction = 4095
        end
        
        image_matrix[pix_ind] = round(Int64, pix_prediction)
    end

    return image_matrix
end

"""
    Log-Likelihood (camera 1-3)
"""
function likelihood_cam13(
        params::T, 
        image::Array{Float64,2},
        population::Float64,
        cv_matrix::Array{Float64,2},
        cam_ind::Int64;
        n_threads = Threads.nthreads()
    ) where {T <: NamedTuple}
    
    tot_loglik = zeros(Float64, n_threads)
    light_coefficient::Float64 = params.light_amp[cam_ind] * 10^5
    
    δ_x::Float64 = params.psx[cam_ind] * 10^-3
    δ_y::Float64 = params.psy[cam_ind] * 10^-3
    
    μ_x::Float64  = params.algmx[cam_ind] * δ_x
    μ_y::Float64  = params.algmy[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.resx[cam_ind]*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.resy[cam_ind]*δ_y).^2) # \sigma x is the same for both
    
    max_pred_amp::Int64 = size(cv_matrix)[2]-1
    
    Threads.@threads for t = 1:n_threads
        
        cum_log_lik = zero(Float64)
        
        for pix_ind in CartesianIndices(image)[t:n_threads:length(image)] 
            if !isnan(image[pix_ind])
                x_edge::Float64 = pix_ind.I[1] * δ_x
                y_edge::Float64 = pix_ind.I[2] * δ_y

                pix_prediction::Float64 = cdf(Normal(μ_x,σ_x), x_edge) - cdf(Normal(μ_x,σ_x), x_edge - δ_x)
                pix_prediction *= cdf(Normal(μ_y,σ_y), y_edge) - cdf(Normal(μ_y,σ_y), y_edge - δ_y)

                pix_prediction = pix_prediction*light_coefficient

                cv_index = floor(Int64, pix_prediction)

                if cv_index > max_pred_amp
                    cv_index = max_pred_amp
                end

                cum_log_lik += cv_matrix[Int64(image[pix_ind]+1), cv_index+1]
            end
        end
        
        tot_loglik[t] = cum_log_lik
        
    end

    return sum(tot_loglik)
end

"""
    Log-Likelihood (camera 4)
"""
function likelihood_cam4(
        params::T, 
        image::Array{Float64,2},
        population::Float64,
        cam_ind::Int64;
        n_threads = Threads.nthreads()
    ) where {T <: NamedTuple}
   
    tot_loglik = zeros(Float64, n_threads)    
    light_coefficient::Float64 = params.cam4_light_amp * 10^5
    
    δ_x::Float64 = params.cam4_psx * 10^-3
    δ_y::Float64 = params.cam4_psy * 10^-3
    
    μ_x::Float64  = params.algmx[cam_ind] * δ_x
    μ_y::Float64  = params.algmy[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.cam4_resx*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.cam4_resy*δ_y).^2)
    
    Threads.@threads for t = 1:n_threads
        
        cum_log_lik = zero(Float64)
        
        for pix_ind in CartesianIndices(image)[t:n_threads:length(image)] 
            if !isnan(image[pix_ind])
                x_edge::Float64 = pix_ind.I[1] * δ_x
                y_edge::Float64 = pix_ind.I[2] * δ_y

                pix_prediction::Float64 = cdf(Normal(μ_x,σ_x), x_edge) - cdf(Normal(μ_x,σ_x), x_edge - δ_x)
                pix_prediction *= cdf(Normal(μ_y,σ_y), y_edge) - cdf(Normal(μ_y,σ_y), y_edge - δ_y)
                pix_prediction = pix_prediction*light_coefficient + params.cam4_ped
                
                if pix_prediction > 10^4
                    pix_prediction = 10^4 # logpdf(truncated(Normal(20000, 2*sqrt(20000)), 0.0, 4096), 4000) gives NaN
                end   
                cum_log_lik += logpdf(truncated(Normal(pix_prediction, params.cam4_light_fluct*sqrt(pix_prediction)), 0.0, 4096), image[pix_ind])
                
                
            end
        end
        
        tot_loglik[t] = cum_log_lik
    end

    return sum(tot_loglik)
end


"""
    Generate simulated event using 4 cameras. 
"""
function generate_event(
        params::D, population::Float64, conv_mat::T; 
        inc_noise=true,
        size = [(70, 70),(70, 70),(40, 40),(70, 70)],
        light_fluctuations = 2.0,
        include_satur = true,
    ) where {T<: NamedTuple, D <: NamedTuple}

    
    img_1 = generate_image_cam13(params, population, conv_mat.cam_1, light_fluctuations, 1, size = size[1], inc_noise=inc_noise, include_satur=include_satur)
    img_2 = generate_image_cam13(params, population, conv_mat.cam_2, light_fluctuations, 2, size = size[2], inc_noise=inc_noise, include_satur=include_satur)
    img_3 = generate_image_cam13(params, population, conv_mat.cam_3, light_fluctuations, 3, size = size[3], inc_noise=inc_noise, include_satur=include_satur)
    img_4 = generate_image_cam4(params, population, 4, size = size[4], inc_noise=inc_noise, include_satur=include_satur)
    
    return (cam_1 = img_1, cam_2 = img_2, cam_3 = img_3, cam_4 = img_4, population = population)
end