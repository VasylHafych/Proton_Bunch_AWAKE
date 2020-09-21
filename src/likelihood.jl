function generate_image(
        params::T, 
        population::Float64,
        cv_matrix::Array{Float64,2},
        light_fluctuations::Float64,
        cam_ind::Int64;
        size::Tuple{Int64, Int64}=(101,101),
        inc_noise = true
    ) where {T <: NamedTuple}
    
    image_matrix = zeros(Int64, size...)
    light_coefficient::Float64 = population*params.int_coeff[cam_ind]
    
    δ_x::Float64 = params.δ_x[cam_ind]
    δ_y::Float64 = params.δ_y[cam_ind]
    
    μ_x::Float64  = params.μ_x[cam_ind] * δ_x
    μ_y::Float64  = params.μ_y[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.s_waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.s_waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.σ_x[cam_ind]*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.σ_y[cam_ind]*δ_y).^2)
    
    bck_cumsum = cumsum(cv_matrix[:,1])
    
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
        
        if pix_prediction > 4095
            pix_prediction = 4095
        end
        
        image_matrix[pix_ind] = round(Int64, pix_prediction)
    end

    return image_matrix
end

function cam_likelihood(
        params::T, 
        image::Array{Int64,2},
        population::Float64,
        cv_matrix::Array{Float64,2},
        cam_ind::Int64;
        n_threads = Threads.nthreads()
    ) where {T <: NamedTuple}
    
    tot_loglik = zeros(Float64, n_threads)
    light_coefficient::Float64 = population*params.int_coeff[cam_ind]
    
    δ_x::Float64 = params.δ_x[cam_ind]
    δ_y::Float64 = params.δ_y[cam_ind]
    
    # To Be Deleted: 
#     if δ_x < 0 || δ_y < 0 || params.tr_size[1] < 0.1 || params.tr_size[2] < 0.1 
#         return -Inf
#     end
    
    μ_x::Float64  = params.μ_x[cam_ind] * δ_x
    μ_y::Float64  = params.μ_y[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.s_waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.s_waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.σ_x[cam_ind]*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.σ_y[cam_ind]*δ_y).^2) # \sigma x is the same for both
    
    max_pred_amp::Int64 = size(cv_matrix)[2]-1
    
    Threads.@threads for t = 1:n_threads
        
        cum_log_lik = zero(Float64)
        
        for pix_ind in CartesianIndices(image)[t:n_threads:length(image)] 

            x_edge::Float64 = pix_ind.I[1] * δ_x
            y_edge::Float64 = pix_ind.I[2] * δ_y

            pix_prediction::Float64 = cdf(Normal(μ_x,σ_x), x_edge) - cdf(Normal(μ_x,σ_x), x_edge - δ_x)
            pix_prediction *= cdf(Normal(μ_y,σ_y), y_edge) - cdf(Normal(μ_y,σ_y), y_edge - δ_y)

            pix_prediction = pix_prediction*light_coefficient

            cv_index = floor(Int64, pix_prediction)

            if cv_index > max_pred_amp
                cv_index = max_pred_amp
            end

            cum_log_lik += log(cv_matrix[image[pix_ind]+1, cv_index+1])
        end
        
        tot_loglik[t] = cum_log_lik
        
    end

    return sum(tot_loglik)
end


function cam_likelihood_debug(
        params::T, 
        image::Array{Int64,2},
        population::Float64,
        cv_matrix::Array{Float64,2},
        cam_ind::Int64;
    ) where {T <: NamedTuple}
    
    image_matrix = zeros(Float64, size(image)...)
    light_coefficient::Float64 = population*params.int_coeff[cam_ind]
    
    δ_x::Float64 = params.δ_x[cam_ind]
    δ_y::Float64 = params.δ_y[cam_ind]
    
    μ_x::Float64  = params.μ_x[cam_ind] * δ_x
    μ_y::Float64  = params.μ_y[cam_ind] * δ_y
    
    σ_x::Float64 = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.s_waist[1] - params.s_cam[cam_ind])^2) 
    σ_y::Float64 = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.s_waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x = sqrt(σ_x^2 + (params.σ_x[cam_ind]*δ_x).^2)
    σ_y = sqrt(σ_y^2 + (params.σ_y[cam_ind]*δ_y).^2)
    
    max_pred_amp::Int64 = size(cv_matrix)[2]-1

    for pix_ind in CartesianIndices(image)

        x_edge::Float64 = pix_ind.I[1] * δ_x
        y_edge::Float64 = pix_ind.I[2] * δ_y

        pix_prediction::Float64 = cdf(Normal(μ_x,σ_x), x_edge) - cdf(Normal(μ_x,σ_x), x_edge - δ_x)
        pix_prediction *= cdf(Normal(μ_y,σ_y), y_edge) - cdf(Normal(μ_y,σ_y), y_edge - δ_y)

        pix_prediction = pix_prediction*light_coefficient

        cv_index = floor(Int64, pix_prediction)

        if cv_index > max_pred_amp
            cv_index = max_pred_amp
        end

        image_matrix[pix_ind] += log(cv_matrix[image[pix_ind]+1, cv_index+1])
    end

    return image_matrix
end

