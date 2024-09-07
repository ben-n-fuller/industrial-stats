module DesignInitializer

using LinearAlgebra
using Random

include("../util/util.jl")
using .Util

include("./model_builder.jl")
using .ModelBuilder

# Fill an nxNxK matrix with values sampled from a uniform dist on [lower, upper]
function init_design(N, K; n=1, lower=-1, upper=1, rng=Random.GLOBAL_RNG)
    lower .+ rand(rng, n, N, K) .* (upper - lower)
end

# Fill an nxNxK matrix with random values ensuring each row sums to 1
function init_mixture_design(N, K; n=1)
    designs = init_design(N, K, n = n, lower=0, upper=1)
    designs ./= sum(designs, dims=3)
    designs
end

function fill_invalid!(X, model_builder, init)
    _, N, K = size(X)

    M = (ModelBuilder.expand ∘ model_builder)(X)

    check_invalid = (x) -> rank(x) < K
    invalids = mapslices(check_invalid, M, dims=[2,3])
    invalid_indices = findall(Util.squeeze(invalids))

    # If no invalid designs, return
    if length(invalid_indices) == 0
        return X
    end

    # Replace invalid designs with new ones in-place
    X[invalid_indices, :, :] = init(N, K, n = length(invalid_indices))

    # Recursively fill invalid designs
    return fill_invalid!(X, model_builder, init)
end

function init_filtered_design(N, K, model_builder; n = 1, init = init_design)
    # Initialize designs
    designs = init(N, K, n = n)

    # Filter out invalid designs
    fill_invalid!(designs, model_builder, init)
    return designs
end

function initialize(N, K, model_builder; n = 1, type="uniform")
    if type == "uniform"
        return init_filtered_design(N, K, model_builder, n = n, init = init_design)
    elseif type == "mixture"
        return init_filtered_design(N, K, model_builder, n = n, init = init_mixture_design)
    else
        throw(ArgumentError("Initializer not recognized. Use 'uniform' or 'mixture'."))
    end
end

function initializer(N, K, model_builder; type="uniform")
    return (n) -> initialize(N, K, model_builder, n = n, type=type)
end

export initialize, initializer

end