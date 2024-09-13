module Geom

export check_polyhedron_consistency, compute_centroid, simplex_max_d, simplex_vec_outer_d_mat, hyper_polyhedron_confine

using Iterators

function check_polyhedron_consistency(a, b)
    # this function checks if the polyhedron is
    #  a.) empty
    #  b.) consistent
    #  c.) inconsistent
    #    (i) and if so bounds are adjusted to be consistent.
    a_plus = sum(a)
    b_plus = sum(b)
    n      = length(a)
    notempty  = (a_plus <= 1) & (b_plus >= 1)
    #=
    println("------------------------------------------------|")
    println(" CHECKING HYPRPOLYHEDRON CONSISTENCY            |")
    println("                                                |")
    println("                                                |")
    =#
    if ~notempty
        println("------------------------------------------------|")
        error("(checkPolyhedronConsistency): polyhedron is empty, check constraints!")
    else
        Ranges = round.(b - a, digits = 6)
        RU     = round(sum(b) - 1, digits = 6)
        RL     = round(1 - sum(a), digits = 6)
        au     = any(Ranges .> RU)
        al     = any(Ranges .> RL)
        notconsistent = al | au

        if ~notconsistent
            #=
            println("polyhedron is consistent, no adjustment made.   |")
            println("------------------------------------------------|")
            println("")
            println("")
            println("") =#
            return a, b
        else
            a_star = Vector{Float64}(undef, n)
            b_star = Vector{Float64}(undef, n)
            for i in 1:n
                a_star[i] = round(max(a[i], 1 - b_plus + b[i]), digits = 6)
                b_star[i] = round(min(b[i], 1 - a_plus + a[i]), digits = 6)
            end
            #=
            println("polyhedron is inconsistent ---------------------|")
            println("LB adjusted to:                                 |")
            println(a_star)
            println("UB adjusted to:                                 |")
            println(b_star)
            println("------------------------------------------------|")
            println("")
            println("")
            =#
            return transpose(a_star), transpose(b_star)
        end
    end
end

function  compute_centroid(a,b)
    ## McLean and Anderson algorithm
    #  only implemented for 3 and 4 factors
    #  Cornell 2011 pp 123
    q = length(a)
    data = vcat(a,b)
    function fitConstr(V, a, b)
        t1 = V .>= a
        t2 = V .<= b
        t3 = all(t1) & all(t2)
        return t3
    end
    if q == 3
        ct_12    = collect(Iterators.product(data[:,1], data[:,2]))
        ct_13    = collect(Iterators.product(data[:,1], data[:,3]))
        ct_23    = collect(Iterators.product(data[:,2], data[:,3]))
        poss_comb = Array{Any}(undef, q*2^(q-1))
        for i in 1:4
            ct = ct_12[i]
            mt = 1-sum(ct)
            poss_comb[i] = [ct[1] ct[2]  round(mt, digits = 6)]
        end
        for i in 1:4
            ct = ct_13[i]
            mt = 1-sum(ct)
            poss_comb[i + 4] = [ct[1]  round(mt, digits = 6) ct[2]]
        end
        for i in 1:4
            ct = ct_23[i]
            mt = 1-sum(ct)
            poss_comb[i + 8] = [round(mt, digits = 6) ct[1] ct[2] ]
        end
        good_ones = falses(length(poss_comb))
        for i in 1:length(poss_comb)
            good_ones[i] = fitConstr(poss_comb[i], a, b)
        end

        comb  = poss_comb[good_ones]
        combf = unique(comb)
        # you have the vertices now put into matrix and average
        vM = Matrix{Float64}(undef, (length(combf), q))
        for i in 1:length(combf)
            vM[i,:] = combf[i]
        end
        centroid = mean(vM, dims = 1)
    elseif q == 4
        ct_123    = collect(Iterators.product(data[:,1], data[:,2], data[:,3]))
        ct_124    = collect(Iterators.product(data[:,1], data[:,2], data[:,4]))
        ct_134    = collect(Iterators.product(data[:,1], data[:,3], data[:,4]))
        ct_234    = collect(Iterators.product(data[:,2], data[:,3], data[:,4]))

        poss_comb = Array{Any}(undef, q*2^(q-1))

        for i in 1:8
            ct = ct_123[i]
            mt = 1-sum(ct)
            poss_comb[i] = [ct[1] ct[2] ct[3] round(mt, digits = 6)]
        end
        for i in 1:8
            ct = ct_124[i]
            mt = 1-sum(ct)
            poss_comb[i + 8] = [ct[1] ct[2] round(mt, digits = 6) ct[3]]
        end
        for i in 1:8
            ct = ct_134[i]
            mt = 1-sum(ct)
            poss_comb[i + 16] = [ct[1] round(mt, digits = 6) ct[2] ct[3] ]
        end
        for i in 1:8
            ct = ct_234[i]
            mt = 1-sum(ct)
            poss_comb[i + 24] = [round(mt, digits = 6) ct[1] ct[2] ct[3] ]
        end

        good_ones = falses(length(poss_comb))
        for i in 1:length(poss_comb)
            good_ones[i] = fitConstr(poss_comb[i], a, b)
        end

        comb = poss_comb[good_ones]
        combf = unique(comb)
        # you have the vertices now put into matrix and average
        vM = Matrix{Float64}(undef, (length(combf), q))
        for i in 1:length(combf)
            vM[i,:] = combf[i]
        end

        centroid = mean(vM, dims = 1)
    elseif q == 5
        error("(compPolyVert) vertices algorithm not implemented for K = 5 factors yet!")
    end
    #=
    println("----------------------------------------------------------|")
    println("  GEOMETRIC CENTROID =                                    |")
    println(round.(centroid, digits = 4))
    println("                                                          |")
    println("  VERTICES =                                              |")
    println(DataFrame(round.(vM, digits = 4)))
    println("                                                          |")
    println("----------------------------------------------------------|")
    println("")
    println("")
    =#
    return centroid, vM
end


# JoBo's alg to confine particles
function hyper_polyhedron_confine(badmix, centroid, a, b)
    #println(badmix)
    vl = any(badmix .< a)
    vu = any(badmix .> b)
    violated = vl | vu
    if ~violated
        newmix = badmix
    else

        while violated
            K       = length(centroid)
            # Initiate new mixture point
            newmix  = transpose(Vector{Float64}(undef, K))
            M_delta = Vector{Float64}(undef, K)
            UorB    = Vector{Float64}(undef, K)
            # ranges
            R       = b - a
            L_delta = (a - badmix) ./ R
            U_delta = (badmix - b) ./ R
            for i in 1:K
                M_delta[i] = max(L_delta[i], U_delta[i])
                if M_delta[i] == U_delta[i]
                    UorB[i] = 1
                elseif M_delta[i] == L_delta[i]
                    UorB[i] = -1
                end
            end
            max_comp   = maximum(M_delta)
            ind_max    = argmax(M_delta)
            ind_s      = collect(1:1:K)
            ind_notmax = setdiff(ind_s, ind_max)

            if UorB[ind_max] == 1
                beta = (b[ind_max] - centroid[ind_max])/(badmix[ind_max] - centroid[ind_max])
                newmix[ind_max] = b[ind_max]
            elseif UorB[ind_max] == -1
                beta = (a[ind_max] - centroid[ind_max])/(badmix[ind_max] - centroid[ind_max])
                newmix[ind_max] = a[ind_max]
            end
            newmix[ind_notmax] .= beta .* badmix[ind_notmax] .+ (1 - beta) .* centroid[ind_notmax]
            # check if the adjustment solved the boundary violation
            vl = any(newmix .< a)
            vu = any(newmix .> b)
            violated = vl | vu
            if violated
                badmix = newmix
            end
        end
    end
    vl2 = any(newmix .< a)
    vu2 = any(newmix .> b)
    violated2 = vl2 | vu2
    if violated2
        println(newmix)
        error("(hyperPolyhedronConfine) WARNING! constrained point is not in the hyperpolyhedron!")
    end
    return newmix
end

end # module


