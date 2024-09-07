module CEXCH
using LinearAlgebra
using ThreadsX

include("../model_builder/model_builder.jl")
using .ModelBuilder

include("../model_builder/design_initializer.jl")
using .DesignInitializer

include("../utility/util.jl")
using .Util

function exchange(X, row, x)
    Xt = copy(X)
    Xt[row, :] = x
    Xt
end

function remove_small_terms(X; tol=1e-6)
    indices = findall(x -> abs(x) < tol, X)
    X[indices] .= 0
    return X
end

function cexch_optimize(X::Matrix{Float64}, obj_crit::Function; max_iters=1000, num_samples=100)
    N, K = size(X)

    # Generate simplex coordinates using identity matrix
    # The kth column corresponds with the simplex vertex for the kth factor
    simplex_coords = I(K)

    # Sample points along line
    sample_points = range(0.0, 1.0, length=num_samples)

    # Initialize objective value
    best_score = obj_crit(X)

    # Pre-allocate memory for designs
    new_designs = zeros(num_samples, N, K)

    # Initialize metadata
    cexch_meta = zeros(4)
    cexch_meta[2] = max_iters
    cexch_meta[3] = num_samples

    # Iterate until no improvement is made
    for iter in 1:max_iters
        improvement = false

        for coord in CartesianIndices(X)
            row, col = coord[1], coord[2]
            # Get the current simplex vertex
            v = simplex_coords[:, col]

            # Get the direction vector
            d = v - X[row, :]
            
            # Generate candidate designs
            for (i, t) in enumerate(sample_points)
                new_designs[i, :, :] = exchange(X, row, X[row, :] + t * d)
            end

            # Compute scores
            scores = obj_crit(new_designs)
            score_opt, i_opt = findmin(Util.squeeze(scores))

            # Update the design matrix and objective value if improvement is found
            if score_opt < best_score
                best_score = score_opt
                X = new_designs[i_opt, :, :]
                improvement = true
            end
        end

        cexch_meta[1] = iter
        if !improvement
            break
        end
    end

    cexch_meta[4] = best_score
    return X, cexch_meta
end

# Simple coordinate exchange algorithm implementation for mixture designs
function cexch!(X::Array{Float64, 3}, obj_crit::Function; max_iters=1000, num_samples=1000)
    n = size(X, 1) # number of design matrices
    meta = zeros(n, 4) # metadata for each design matrix

    # Parallelize over initializations
    Threads.@threads for i in 1:n
        X_new, m = cexch_optimize(X[i, :, :], obj_crit, max_iters=max_iters, num_samples=num_samples)
        X[i, :, :] = X_new
        meta[i, :] = m
    end

    return remove_small_terms(X), meta
end

# Simple coordinate exchange algorithm implementation for mixture designs
function cexch(X::Array{Float64, 3}, obj_crit::Function; max_iters=1000, num_samples=1000)
    X = copy(X)
    cexch!(X, obj_crit, max_iters=max_iters, num_samples=num_samples)
end

export cexch!, cexch

end