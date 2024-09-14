module Simplex

export add, multiply, norm, vec_dist, vec_outer_d_mat, max_d, scale

function add(p, q)
    eprod = p .* q
    result = eprod ./ sum(eprod)
    # just in case any values are 0
    smallest = eps()^3
    ind_temp = result .< smallest
    result[ind_temp] .= smallest
    result = result ./ sum(result)
    return result
end

function multiply(p, lambda)
    eexp = p .^ lambda
    result = eexp ./ sum(eexp)
    # just in case any values are 0
    smallest = eps()^3
    ind_temp = result .< smallest
    result[ind_temp] .= smallest
    result = result ./ sum(result)
    return result
end

function norm(p)
    K = length(p)
    Mt = Matrix{Float64}(undef, K, K)
    for i in 1:K
        for j in 1:K
            Mt[i,j] = ( log( p[i] / p[j] ) )^2
        end
    end
    result = sqrt(sum(Mt)/(2*K))
    return result
end

function vec_dist(p, q)
    K = length(p)
    if K != length(q)
        error("simplexVecDist: vectors must be same length!")
    end
    Mt = Matrix{Float64}(undef, K, K)
    for i in 1:K
        for j in 1:K
            Mt[i,j] = ( log( p[i] / p[j] )  - log( q[i] / q[j] ))^2
        end
    end
    result = sqrt(sum(Mt)/(2*K))
    return result
end

function vec_outer_d_mat(V)
    nr, nc = size(V)
    D      = Matrix{Float64}(undef, nr, nr)
    for i in 1:nr
        vt = V[i,:]
        bad_ind = vt .== 0
        vt[bad_ind] .= 1e-4
        V[i,:] = vt/sum(vt)
    end

    for i in 1:nr
        for j in 1:nr
            D[i, j] = vec_dist(V[i,:], V[j,:])
        end
    end
    return D
end

function max_d(K)
    t1 = fill(0.0001, K-1)
    t2 = 1 - sum(t1)
    t3 = vcat(t1, t2)
    ct = fill(1/K,K)
    result = vec_dist(t3,ct)
    return result
end

function scale(p, L)
    np = norm(p)
    ct = L/np
    result = multiply(p, ct)
    return result
end

end # module