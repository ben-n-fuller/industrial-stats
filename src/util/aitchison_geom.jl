module AitchisonGeometry

# Perturbation takes the element-wise product of two vectors and normalizes
function aitch_perturb(x::Vector{Float64}, y::Vector{Float64})::Vector{Float64}
    x .* y ./ sum(x .* y)
end

# Power operation raises each element in the vector to the scalar and normalizes
function aitch_power(x::Vector{Float64}, a::Float64)::Vector{Float64}
    x .^ a ./ sum(x .^ a)
end

# Inner product of two vectors
function aitch_inner(x::Vector{Float64}, y::Vector{Float64})::Float64
    # Compute the matrices of pairwise log ratios for x and y
    log_ratios_x = log.(x) .- log.(x')
    log_ratios_y = log.(y) .- log.(y')

    # Compute the inner product
    1 / (2 * length(x)) * (sum(log_ratios_x .- log_ratios_y) .^ 2)
end

export aitch_perturb, aitch_power, aitch_inner

end