module MCIntegration

export mc_integrate
export compute_volume

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

include("./tensor_ops.jl")
using .TensorOps

include("./sampler.jl")
import .Sampler

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

end