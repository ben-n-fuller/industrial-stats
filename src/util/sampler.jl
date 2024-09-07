module Sampler

export sample_hypercube, sample_simplex, rejection_sample
export get_simplex_constraints

# External libs
using LinearAlgebra
using Distributions
using Polyhedra
using CDDLib
using Statistics
using Random

# Internal modules
include("../model/model_builder.jl")
import .ModelBuilder

include("../model/design_initializer.jl")
import .DesignInitializer

include("./util.jl")
using .Util# Sample points from the simplex using the Dirichlet distribution

function sample_simplex(N, K; rng=Random.GLOBAL_RNG)
    a = ones(K)
    sampler = Dirichlet(a)
    samples = zeros(N, K)

    for i in axes(samples, 1)
        samples[i, :] .= rand(rng, sampler)
    end

    return samples
end

# Sample n points from the hypercube [-lower, lower]^K
function sample_hypercube(n, K; lower=-1, upper=1, rng=Random.GLOBAL_RNG)
    squeeze(DesignInitializer.init_design(n, K; rng=rng, lower=lower, upper=upper))
end

# Get default linear constraints for the simplex
function get_simplex_constraints(K)
    A = [
        -1 * I(K);
        ones(1, K)
    ]
    b = zeros(K + 1)
    b[end] = 1
    return A, b
end

# Rejection sampler for a polytope
function rejection_sample(n, K, A, b, sampler)
    X = sampler(n, K)
    satisfies = (x) -> all(A * x .<= b)

    while true
        # Get the design points in X that satisfy the constraints
        good_points = vec(mapslices(satisfies, X; dims=2))
        num_bad_points = sum(.!good_points)

        # If there are no bad points, return the good points
        if num_bad_points == 0
            return X
        end

        # Resample bad points
        X[.!good_points, :] .= sampler(num_bad_points, K)
    end
end

end # module
