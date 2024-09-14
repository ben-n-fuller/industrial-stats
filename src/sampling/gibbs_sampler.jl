module GibbsSampler

export sample_truncated_dirichlet

using Distributions

function sample_truncated_dirichlet(N, a, b, gamma, centroid)

    n        = length(a)
    n2       = length(b)
    if ~(n == n2)
        error("(rTDirichlet): lower and upper bound vectors are not the same length!")
    end
    # set Dirichlet parameters to uniform if not specified
    if gamma == undef
        gamma = fill(1, n)
    end
    data      = Matrix{Float64}(undef, N + 1, n)
    data[1,:] = centroid
    for j in 2:(N + 1)
        for i in 1:(n - 1)
            if i == 1
                var_ind = fill(j - 1, n - 1)
            elseif (i > 1) & (i <= n - 1)
                var_ind = vcat(fill(j, i - 1), fill(j - 1, (n - 1)  - (i - 1)))
            end
            x_mi_t = Vector{Float64}(undef, n-1)
            for k in 1:(n - 1)
                x_mi_t[k] = data[var_ind[k], k]
            end

            xi_star_t  = xiStar(x_mi_t, i)
            alpha_i_t  = falpha_i(i, a, b, xi_star_t, n)
            beta_i_t   = fbeta_i(i, a, b, xi_star_t, n)
            U          = Uniform(0,1)
            ut         = rand(U, 1)[1]
            yi_g_xmi   = uToTbeta(ut, gamma[i], gamma[n], alpha_i_t, beta_i_t)
            data[j, i] = xi_star_t * yi_g_xmi
        end
        data[j, n] = 1 - sum(data[j, 1:(n-1)])
    end
    data = data[2:(N + 1),:]
    checkRanges(data, a, b)
    return data
end

function xiStar(x, i)
    result =  1 - sum(deleteat!(x, i))
    return result
end

function falpha_i(i, a, b, xi_star, n)
    result = max( a[i]/xi_star, 1 - b[n]/xi_star)
    return result
end

function fbeta_i(i, a, b, xi_star, n)
    result = min( b[i]/xi_star, 1 - a[n]/xi_star)
end

function uToTbeta(u, s1, s2, ac, bc)
    Beta_t   = Beta(s1, s2)
    t1       = u * (cdf(Beta_t, bc) - cdf(Beta_t, ac))
    t2       = cdf(Beta_t, ac) + t1
    result   = quantile(Beta_t, t2)
    return result
end

function checkRanges(data, a, b)
    K = length(a)
    tt = BitArray(fill(0,K))
    for i in 1:K
        tt[i] = all(data[:,i] .>= a[i]) & all(data[:,i] .<= b[i])
    end
    result = all(tt)
    if !result
        println("warning: extreme values of alpha may cause simulants outside of the polyhedron!")
    end
end

end