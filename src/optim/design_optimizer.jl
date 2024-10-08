module DesignOptimizer

using SpecialFunctions
using LinearAlgebra
using ThreadsX

function obj_crit_line(t, X, row, d, obj_crit)
    # Compute the candidate design point
    x_t = X[row, :] + t * d

    # Temporarily update the design matrix for evaluation
    old_row = copy(X[row, :])
    X[row, :] = x_t

    # Evaluate the objective function
    score = obj_crit(X)

    # Restore the original design matrix
    X[row, :] = old_row

    score
end

function jl_optimizer(X, row, d, obj_crit)
    # Perform optimization along the line segment
    opt_func = (t) -> obj_crit_line(t, X, row, d, obj_crit)
    opt_result = optimize(opt_func, 0.0, 1.0)
    t_opt = Optim.minimizer(opt_result)
    score_opt = Optim.minimum(opt_result)

    return (score_opt, t_opt)
end

function m_optimizer(X, row, d, obj_crit; n_samples=1000)
    # Sample uniformly spaced points along the line segment
    t_samples = range(0.0, 1.0, length=n_samples)

    # Compute score for each candidate design point X[row, :] + t * d
    scores = zeros(n_samples)
    for (i, t) in enumerate(t_samples)
        score = obj_crit_line(t, X, row, d, obj_crit)
        scores[i] = score
    end

    # Find the best score and corresponding t
    score_opt, i_opt = findmin(scores)
    t_opt = t_samples[i_opt]

    return (score_opt, t_opt)
end

end