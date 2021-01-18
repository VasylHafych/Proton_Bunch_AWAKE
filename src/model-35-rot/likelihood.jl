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
    
    x_edges = range(0, length = Base.size(image_matrix)[2]+1, step=δ_x) 
    y_edges = range(0, length = Base.size(image_matrix)[1]+1, step=δ_y) 
    
    rot_mat = [cos(params.α) sin(params.α) ; -sin(params.α) cos(params.α)]
    mvdist = MvNormal([μ_x, μ_y], [σ_x^2 0; 0 σ_y^2])
    int_func(x) = pdf(mvdist, rot_mat*x)
    mvmat = [int_func([x,y]) for y in y_edges, x in x_edges]  
     
    bck_cumsum = cumsum(exp.(cv_matrix[:,1]))
    
    for pix_ind in CartesianIndices(image_matrix)
    
        @inbounds x_edge = pix_ind.I[2] 
        @inbounds y_edge = pix_ind.I[1] 

#                 pix_prediction = hcubature(int_func, [x_edges[x_edge], y_edges[y_edge]], [x_edges[x_edge+1], y_edges[y_edge+1]], rtol=0.1, maxevals=5)[1]

        pix_prediction = nanmean([mvmat[x_edge:x_edge+1, y_edge:y_edge+1]...])*δ_x*δ_y
        pix_prediction = pix_prediction*light_coefficient
        
        if inc_noise
            
            # CV matrix
            # to try: round / floor
            pix_prediction = argmin(abs.(cumsum(exp.(cv_matrix[:, round(Int64, pix_prediction)+1])) .- rand())) - 1
            
            # Used to generate evenl 
#             pix_prediction = rand(truncated(Normal(pix_prediction, 0.5+light_fluctuations*sqrt(pix_prediction)), 0, 4096))
#             background_tmp = bck_cumsum .- rand()
#             background_tmp[background_tmp .< 0 ] .= Inf
#             pix_prediction += argmin(background_tmp) - 1
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
    
    x_edges = range(0, length = Base.size(image_matrix)[2]+1, step=δ_x) 
    y_edges = range(0, length = Base.size(image_matrix)[1]+1, step=δ_y) 
    
    rot_mat = [cos(params.α) sin(params.α) ; -sin(params.α) cos(params.α)]
    mvdist = MvNormal([μ_x, μ_y], [σ_x^2 0; 0 σ_y^2])
    int_func(x) = pdf(mvdist, rot_mat*x)
    mvmat = [int_func([x,y]) for y in y_edges, x in x_edges]  

    for pix_ind in CartesianIndices(image_matrix)
    
        @inbounds x_edge = pix_ind.I[2] 
        @inbounds y_edge = pix_ind.I[1] 

        pix_prediction = nanmean([mvmat[x_edge:x_edge+1, y_edge:y_edge+1]...])*δ_x*δ_y

#       pix_prediction = hcubature(int_func, [x_edges[x_edge], y_edges[y_edge]], [x_edges[x_edge+1], y_edges[y_edge+1]], rtol=0.1, maxevals=5)[1]

        if pix_prediction < 0 || isnan(pix_prediction)
            pix_prediction = 0.0
        end

        pix_prediction = pix_prediction*light_coefficient + params.cam4_ped
        
        if inc_noise
            pix_prediction = rand(truncated(Normal(pix_prediction, params.cam4_light_fluct*sqrt(pix_prediction)), 0.0, 4095)) 
#             pix_prediction = rand(truncated(Poisson(pix_prediction), 0.0, 4095)) 
        end
        
        if include_satur && pix_prediction > 4095
            pix_prediction = 4095
        end
        
        image_matrix[pix_ind] = round(Int64, pix_prediction) # round or floor
    end

    return image_matrix
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


function likelihood_cam4(
        params::NamedTuple, 
        image::Array{F,2},
        population::AbstractFloat,
        cam_ind::Integer;
        n_threads::Integer = Threads.nthreads()
    ) where {F <: AbstractFloat}
   

    VT = eltype(params.tr_size)
    tot_loglik::Array{VT} = zeros(VT, n_threads)    
    light_coefficient::VT = params.cam4_light_amp * 10^5
    
    δ_x::VT = params.cam4_psx * 10^-3
    δ_y::VT = params.cam4_psy * 10^-3
    
    @inbounds μ_x::VT  = params.algmx[cam_ind] * δ_x
    @inbounds μ_y::VT  = params.algmy[cam_ind] * δ_y
    
    @inbounds σ_x_1::VT = sqrt(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - params.s_cam[cam_ind])^2)
    @inbounds σ_y_1::VT = sqrt(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - params.s_cam[cam_ind])^2)
    
    σ_x::VT = sqrt(σ_x_1^2 + (params.cam4_resx*δ_x)^2)
    σ_y::VT = sqrt(σ_y_1^2 + (params.cam4_resy*δ_y)^2)
    
    x_edges = range(0, length = Base.size(image)[2]+1, step=δ_x) 
    y_edges = range(0, length = Base.size(image)[1]+1, step=δ_y)   
    
    rot_mat = [cos(params.α) sin(params.α) ; -sin(params.α) cos(params.α)]
    mvdist = MvNormal([μ_x, μ_y], [σ_x^2 0; 0 σ_y^2])
    int_func(x) = pdf(mvdist, rot_mat*x)
    mvmat = [int_func([x,y]) for y in y_edges, x in x_edges]  
    
    Threads.@threads for t in eachindex(tot_loglik)
        
        cum_log_lik = zero(Float64) 
        
        @inbounds for pix_ind in CartesianIndices(image)[t:n_threads:length(image)] 
            @inbounds if !isnan(image[pix_ind])
                
                @inbounds x_edge = pix_ind.I[2] 
                @inbounds y_edge = pix_ind.I[1] 
                
                pix_prediction = nanmean([mvmat[x_edge:x_edge+1, y_edge:y_edge+1]...])*δ_x*δ_y

#                 pix_prediction = hcubature(int_func, [x_edges[x_edge], y_edges[y_edge]], [x_edges[x_edge+1], y_edges[y_edge+1]], rtol=0.1, maxevals=5)[1]
                
                if pix_prediction < 0 || isnan(pix_prediction)
                    pix_prediction = 0.0
                end
                
                pix_prediction = pix_prediction*light_coefficient + params.cam4_ped
                
                @inbounds cum_log_lik += logpdf(truncated(Normal(pix_prediction, params.cam4_light_fluct*sqrt(pix_prediction)), 0.0, 4096.0), image[pix_ind]) # leads to -Inf
#                 @inbounds cum_log_lik += logpdf(Normal(pix_prediction, params.cam4_light_fluct*sqrt(pix_prediction)), image[pix_ind]) # significantly speeds up auto diff
                
            end
        end
        
        @inbounds tot_loglik[t] = cum_log_lik
    end
    return sum(tot_loglik)
end


function likelihood_cam13(
        params::NamedTuple, 
        image::Array{F,2},
        population::AbstractFloat,
        cv_matrix::Array{C,2},
        cv_func::Function, 
        cam_ind::Integer;
        n_threads::Integer = Threads.nthreads()
    ) where {F <: AbstractFloat, C <: AbstractFloat}
    
    VT = eltype(params.tr_size)
    tot_loglik::Array{VT} = zeros(VT, n_threads)    
    
    light_coefficient::VT = params.light_amp[cam_ind] * 10^5
    
    δ_x::VT = params.psx[cam_ind] * 10^-3
    δ_y::VT = params.psy[cam_ind] * 10^-3
    
    @inbounds μ_x::VT  = params.algmx[cam_ind] * δ_x
    @inbounds μ_y::VT  = params.algmy[cam_ind] * δ_y
    
    @inbounds σ_x_1::VT = sqrt.(params.tr_size[1]^2 + 10^-4*params.ang_spr[1]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    @inbounds σ_y_1::VT = sqrt.(params.tr_size[2]^2 + 10^-4*params.ang_spr[2]^2*(params.waist[1] - params.s_cam[cam_ind])^2) 
    
    σ_x::VT = sqrt(σ_x_1^2 + (params.resx[cam_ind]*δ_x)^2)
    σ_y::VT = sqrt(σ_y_1^2 + (params.resy[cam_ind]*δ_y)^2) 
    
    x_edges = range(0, length = Base.size(image)[2]+1, step=δ_x) 
    y_edges = range(0, length = Base.size(image)[1]+1, step=δ_y) 
    
    rot_mat = [cos(params.α) sin(params.α) ; -sin(params.α) cos(params.α)]
    mvdist = MvNormal([μ_x, μ_y], [σ_x^2 0; 0 σ_y^2])
    int_func(x) = pdf(mvdist, rot_mat*x)
    mvmat = [int_func([x,y]) for y in y_edges, x in x_edges]  
    
    max_pred_amp = size(cv_matrix)[2]-1
    
    Threads.@threads for t in eachindex(tot_loglik)
        
        cum_log_lik = zero(Float64)
        
        @inbounds for pix_ind in CartesianIndices(image)[t:n_threads:length(image)] 
            @inbounds if !isnan(image[pix_ind])
                
                @inbounds x_edge = pix_ind.I[2] 
                @inbounds y_edge = pix_ind.I[1] 

#                 pix_prediction = hcubature(int_func, [x_edges[x_edge], y_edges[y_edge]], [x_edges[x_edge+1], y_edges[y_edge+1]], rtol=0.1, maxevals=5)[1]

                pix_prediction = nanmean([mvmat[x_edge:x_edge+1, y_edge:y_edge+1]...])*δ_x*δ_y
                pix_prediction = pix_prediction*light_coefficient

                if pix_prediction > max_pred_amp - 1
                    pix_prediction = max_pred_amp - 1
                elseif pix_prediction < 0.0 || isnan(pix_prediction)
                    pix_prediction = 0.0
                end
                
                @inbounds cum_log_lik += cv_func(cv_matrix, image[pix_ind], pix_prediction)
                
            end
        end
        tot_loglik[t] = cum_log_lik
        
    end

    return sum(tot_loglik)
end

function conv_tabl_discrete(cv_matrix::Array{F,2}, observed::Real, expected::Real) where {F<:AbstractFloat}  
    return cv_matrix[convert(Integer, observed+1), round(Integer, expected+1)]     
end

function conv_tabl_cont(cv_matrix::Array{F,2}, observed::Real, expected::Real) where {F<:AbstractFloat}   
    
    left_exp, right_exp = floor(Integer, expected+1), ceil(Integer, expected+1)
    exp_rem = mod(expected+1, 1)
    left_prob, right_prob = cv_matrix[convert(Integer, observed+1), left_exp], cv_matrix[convert(Integer, observed+1), right_exp]
    int_prob = log(left_prob + (right_prob - left_prob)*exp_rem)
    return int_prob

end

nanmean(x) = mean(filter(!isnan,x))