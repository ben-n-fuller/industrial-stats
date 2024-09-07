module Numeric

# External libs
using LinearAlgebra
using Distributions
using Polyhedra
using CDDLib
using Statistics
using Random

# Internal modules
include("../model_builder/model_builder.jl")
import .ModelBuilder

include("../model_builder/design_initializer.jl")
import .DesignInitializer

include("../utility/util.jl")
using .Util

# Sample points from the simplex using the Dirichlet distribution
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

# Compute the volume of a polytope using the CDD library
function compute_volume(A, b; affines=BitSet([]))
    p = polyhedron(hrep(A, b, affines), CDDLib.Library())
    return volume(p)
end

# Compute the mean of the outer product of the feature expansion function applied to a set of points
function compute_outer_product_mean(X, f)
    expanded = f(X)
    total_sum = zeros((size(expanded, 2), size(expanded, 2)))

    for i in axes(expanded, 1)
        total_sum .+= expanded[i, :, :] * expanded[i, :, :]'
    end

    total_sum ./ size(expanded, 1)
end

# Integrate over a constrained simplex
function mc_integrate_constrained_simplex(A, b, f; n=100_000)
    X = rejection_sample(n, size(A, 2), A, b, sample_simplex)

    # Get augmented constraints with simplex for volume computation
    A_simp, b_simp = get_simplex_constraints(size(A, 2))
    A = vcat(A, A_simp)
    b = vcat(b, b_simp)
    
    # Compute volume of the polytope
    vol = compute_volume(A, b)

    # Compute the integral
    return compute_outer_product_mean(X, f) * vol
end

# Integrate over a constrainted hypercube
function mc_integrate_constrained_hypercube(A, b, f; n=100_000)
    sampler = (n, K) -> squeeze(DesignInitializer.init_design(n, K))
    X = rejection_sample(n, size(A, 2), A, b, sampler)

    # Compute volume of the polytope
    vol = compute_volume(A, b)

    # Compute the integral
    return compute_outer_product_mean(X, f) * vol
end

# Dispatch function for integration over simplex or hypercube
function mc_integrate(A, b, f; n=100_000, mixture=true)
    if mixture
        return mc_integrate_constrained_simplex(A, b, f, n=n)
    else
        return mc_integrate_constrained_hypercube(A, b, f, n=n)
    end
end

export mc_integrate
export compute_volume
export sample_hypercube, sample_simplex, rejection_sample
export get_simplex_constraints

end