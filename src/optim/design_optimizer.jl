module DesignOptimizer

using SpecialFunctions
using LinearAlgebra
using ThreadsX
using LinearAlgebra
using Optim
using ForwardDiff
using Optimization, OptimizationNLopt

function create_univariate_objective(X, row, d, obj_crit)
    og_row = copy(X[row, :])
    function univariate_obj(x)
        X[row, :] .= og_row .+ x * d
        score = obj_crit(X)
        X[row, :] .= og_row
        return score
    end
    return univariate_obj
end

function bounded_univariate_optimizer(X, row, d, obj_crit)
    univariate_obj = create_univariate_objective(X, row, d, obj_crit)
    result = Optim.optimize(univariate_obj, 0, 1.0)
    optim_point = X[row, :] .+ Optim.minimizer(result) * d
    return optim_point, Optim.minimum(result)
end

function bounded_univariate_gradient_optimizer(X, row, d, obj_crit; optimizer=NLopt.LD_BFGS)
    # Make univariate optimizer
    univariate_obj = create_univariate_objective(X, row, d, obj_crit)
    univariate_obj_vec = (x, _) -> univariate_obj(x[1])

    # Initial values and bounds
    x0 = [0]
    p=[0]
    lb = [0]
    ub = [1.0]

    # Define optimization problem
    f = OptimizationFunction(univariate_obj_vec, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(f, x0, p, ub=ub, lb=lb)
    solution = solve(prob, optimizer())
    optim_point = X[row, :] .+ solution.u[1] * d

    # Evaluate
    tmp = copy(X[row, :])
    X[row, :] .= optim_point
    score = obj_crit(X)
    X[row, :] .= tmp

    return optim_point, score
end

function bounded_univariate_gradient_optimizer(optimizer=NLopt.LD_BFGS)
    return (X, row, d, obj_crit) -> bounded_univariate_gradient_optimizer(X, row, d, obj_crit; optimizer=optimizer)
end

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
    opt_result = Optim.optimize(opt_func, 0.0, 1.0)
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