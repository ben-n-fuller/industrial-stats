module CEXCH

export cexch

using LinearAlgebra

function basic_score_comparator(a, b)
    a < b
end

function cexch_optimize(
        X_in::Matrix{Float64}, 
        obj_crit::Function, 
        optimizer::Function; 
        max_num_iters=1000, 
        score_comparator=basic_score_comparator
    )

    # Copy the input design matrix
    X = copy(X_in)

    # Get N and K
    N, K = size(X)

    # Generate simplex coordinates using identity matrix
    # The kth column corresponds with the simplex vertex for the kth factor in an unconstrained setting
    simplex_coords = I(K)

    # Initialize objective value
    best_score = obj_crit(X)

    # Iterate until no improvement is made
    iter = 0
    while iter < max_num_iters
        improvement = false

        for coord in CartesianIndices(X)
            row, col = coord[1], coord[2]

            # Get the current simplex vertex
            v = simplex_coords[:, col]

            # Get the direction vector
            d = v - X[row, :]

            # Optimize the objective function along the direction vector
            optim_point, score_opt = optimizer(X, row, d, obj_crit)

            # Update the design matrix and objective value if improvement is found
            if score_comparator(score_opt, best_score)
                best_score = score_opt
                X[row, :] .= optim_point
                improvement = true
            end
        end

        if !improvement
            break
        end

        iter += 1
    end

    return X, best_score, iter
end

function cexch(
        X::Array{Float64, 3},
        obj_crit::Function, 
        optimizer::Function; 
        max_iters=1000, 
        score_comparator=basic_score_comparator
    )
    
    # Get the number of initial design matrices
    n, N, K = size(X) # number of initial design matrices

    # Initialize arrays to store scores and number of iterations    
    scores = Float64[]
    num_iterations = Int[]
    opt_designs = zeros(n, N, K)

    for i in 1:n
        optimized, best_score, num_iters = cexch_optimize(
            X[i, :, :], 
            obj_crit, 
            optimizer; 
            max_num_iters=max_iters, 
            score_comparator=score_comparator
        )

        push!(scores, best_score)
        push!(num_iterations, num_iters)
        opt_designs[i, :, :] .= optimized
    end

    # return scores, num_iterations    
    return opt_designs, scores, num_iterations
end

end