module ParticleSwarm

export search_simplex

import("../geom/geom.jl")
using .Geom

import("../geom/simplex.jl")
using .Simplex

import("../sampling/gibbs_sampler.jl")
using .GibbsSampler

import("../model/design_initializer.jl")
using .DesignInitializer

using Distributions, LinearAlgebra

function search_simplex(;
    N::Int64,                       # := Number of replicate points in design
    K::Int64,                       # := Number of design factors
    S::Int64,                       # := Number of particles
    objective,                      # := function to be minimized
    order = nothing,
    max_iter     = 1000,
    w            = 1/(2*log(2)),
    c1           = 0.5 + log(2),
    c2           = 0.5 + log(2),
    nn           = 3,
    printProg    = true,
    PSOversion   = "gbest",
    relTol       = 0,
    maxStag      = 500,
    vmaxScale    = 2,
    L            = undef,
    U            = undef)

    ## check if PSO call works
    if ~(PSOversion in ["gbest", "lbest"])
        error("PSOversion must be gbest or lbest")
    end

    no_bounds = (L == undef) & (U == undef)
    if ~no_bounds
        ## investigate the hyperpolytope
        a, b               = Geom.check_polyhedron_consistency(L, U)
        countVEF(a,b)
        centroid, vertices = Geom.compute_centroid(a,b)
        Dmat               = Simplex.simplex_vec_outer_d_mat(vertices)
        maxd               = maximum(Dmat)/(2*vmaxScale)
        use_bounds         = true
    else
        # maxd is particle step size
        # simplexMaxD is from centroid to vertex (too 4 decimals)
        #  so step size is factor 1/vmaxScale of that
        maxd       = Simplex.simplex_max_d(K)/vmaxScale
        a          = undef
        b          = undef
        centroid   = undef
        use_bounds = false
    end



    ## initialize swarm
    X, V, f, p_best, f_pbest, g_best, f_gbest, l_best, f_lbest, l_bestIndex, neighborHoods = initialize_swarm(; N = N, K = K, S = S, objective = objective, order = order, nn = nn, maxd = maxd, use_bounds = use_bounds, a = a, b = b, centroid = centroid)

    ## indicator for no imporovement in g_best
    improvement = false
    ## set stagnation counter
    stagnation = 0
    ## particle iteration set
    set = [1:1:S;]

    ## set iteration
    iter = 1
    if printProg
        print("\n", iter)
    end

    ## reltol convergence
    reltol_conv    = false
    reltol         = eps(Float64)#eps(Float64)^(0.8) #sqrt(eps(Float64))
    pbest_fit_iter = typemax(Float64)


    while iter < max_iter && stagnation < maxStag && ~reltol_conv
        iter += 1
        if printProg
            print("\n", iter)
        end

        if ~improvement
            ## generate Neighborhood communication if no improvement in g_best
            #print("======= New Neighbors =======")
            neighborHoods = gen_neighbors(S, nn)
            stagnation += 1
        else
            #neighborHoods[iter] = neighborHoods[iter - 1]
            stagnation = 0
        end

        X_t           = deepcopy(X)
        V_t           = deepcopy(V)
        f_t           = deepcopy(f)
        p_best_t      = deepcopy(p_best)
        f_pbest_t     = deepcopy(f_pbest)
        g_best_t      = deepcopy(g_best)
        f_gbest_t     = deepcopy(f_gbest)

        if (PSOversion  == "lbest")
            l_best_t        = deepcopy(l_best)
            f_lbest_t       = deepcopy(f_lbest)
            l_bestIndex_t   = deepcopy(l_bestIndex)
            neighborHoods_t = deepcopy(neighborHoods)
        elseif (PSOversion == "gbest")
            l_best_t      = deepcopy(l_best)
            f_lbest_t     = deepcopy(f_lbest)
            l_bestIndex_t = deepcopy(l_bestIndex)
        end

        ## last iterations best value
        pbest_fit_iterprior_t = deepcopy(pbest_fit_iter)
        f_gbest_prev = deepcopy(f_gbest_t)

        ## swarm knowledge update
        pset = shuffle(set)
        for i in pset
            ## update position and velocity
            X_t[:, :, i], V_t[:, :, i] = update_velocity_and_position(X_t[:, :, i], V_t[:, :, i], p_best_t[:, :, i], l_best_t[:,:,i], g_best_t, w, c1, c2, PSOversion, maxd, use_bounds, centroid, a, b)

            ## update fitness
            f_t[i] = objective(X_t[:, :, i], order = order)
        end

        ## swarm knowledge update
        for i in pset
            ## update personal and global update bests
            p_best_t[:, :, i], f_pbest_t[i], g_best_t, f_gbest_t = update_pg_bests(X_t[:, :, i], f_t[i], p_best_t[:, :, i], f_pbest_t[i], g_best_t,  f_gbest_t)
        end

        ## local neighborhood update
        if PSOversion == "lbest"
            for i in pset
                l_best_t[:, :, i], f_lbest_t[i], l_bestIndex_t[i] = update_l_best(p_best_t, f_pbest_t, l_best_t[:,:,i], f_lbest_t[i], l_bestIndex_t[i], neighborHoods_t[i])
            end
        end

        X       = deepcopy(X_t)
        V       = deepcopy(V_t)
        f       = deepcopy(f_t)
        p_best  = deepcopy(p_best_t)
        f_pbest = deepcopy(f_pbest_t)
        g_best  = deepcopy(g_best_t)
        f_gbest = deepcopy(f_gbest_t)

        if PSOversion  == "lbest"
            l_best      = deepcopy(l_best_t)
            f_lbest     = deepcopy(f_lbest_t)
            l_bestIndex = deepcopy(l_bestIndex_t)
        end

        ## iteration checks
        if f_gbest_t == f_gbest_prev
            improvement = false
        else
            improvement = true
        end

        ## reltol checker
        if ~(relTol == 0)
            pbest_fit_iter = minimum(f_t)
            rdiff = abs(pbest_fit_iterprior_t - pbest_fit_iter)
            #println("")
            #println(rdiff)
            if rdiff == 0
                #println("(0)rdiff = ", rdiff)
                reltol_conv = false
            else
                reltol_conv = rdiff <= reltol
                #println("rdiff = ", rdiff)
            end
        end

    end

    return iter, S, f_gbest, g_best

end

function initialize_swarm(; N, K, S, objective, order = nothing, nn, maxd, use_bounds, a, b, centroid)
    ## swarm swarm_initialization
    ## dimensions are
    X0 = Array{Float64}(undef, (N, K, S))
    V0 = deepcopy(X0)
    # populate the particle array and velocity array
    #nInt = S - 10
    if ~use_bounds
        for i in 1:S
            X0[:, :, i] = DesignInitializer.genRandDesign_mix(N, K)
            vt          = DesignInitializer.genRandDesign_mix(N, K)
            nvt         = Simplex.simplex_norm(vt)
            if nvt > maxd
                vt = Simplex.simplex_scale(vt, maxd)
            end
            V0[:, :, i] = vt
        end
    else
        for i in 1:S
            X0[:, :, i] = GibbsSampler.tDirGibbs(N, a, b, undef, centroid)
            vt          = DesignInitializer.genRandDesign_mix(N, K)
            nvt         = Simplex.simplex_norm(vt)
            if nvt > maxd
                vt = Simplex.simplex_scale(vt, maxd)
            end
            V0[:, :, i] = vt
        end
    end

    ## evaluate the objective
    f0 = Vector(undef, S)
    if isnothing(order)
        for i in 1:S
            f0[i] = objective(X0[:, :, i])
        end
    else
        for i in 1:S
            f0[i] = objective(X0[:, :, i], order = order)
        end
    end
    ## personal bests
    p_best0      = deepcopy(X0)
    f_pbest0     = fill(typemax(Float64), S)
    ## global bests
    g_best0      = deepcopy(X0[:, :, 1])
    f_gbest0     = typemax(Float64)
    ## local bests
    neighbors    = gen_neighbors(S, nn)
    l_best0      = deepcopy(X0)
    f_lbest0     = fill(typemax(Float64), S)
    l_bestIndex0 = collect(1:1:S)
    ## local neighborhood update
    for i in collect(1:1:S)
        l_best0[:, :, i], f_lbest0[i], l_bestIndex0[i] = update_l_best(p_best0, f_pbest0, l_best0[:,:,i], f_lbest0[i], l_bestIndex0[i], neighbors[i])
    end

    return X0, V0, f0, p_best0, f_pbest0, g_best0, f_gbest0, l_best0, f_lbest0, l_bestIndex0, neighbors
end

## generate neighborhood communication links
function gen_neighbors(S, nn = 3)
    # NOTE: this function generates a list of length S,
    #       the ith element is the list of neighbors of particle i
    p_avg = 1 - (1-1/S)^nn
    d     = Uniform(0,1)
    tmp   = reshape(rand(d, S*S) .< p_avg, (S, S))
    # connect particles to themselves
    tmp[diagind(tmp)] .= 1

    Neighbors = Array{Any}(undef, S)
    for i in 1:S
        indtemp  = findall(tmp[:, i])
        l        = length(indtemp)
        Neighbors[i] = indtemp
    end
    return Neighbors
end

function update_velocity_and_position(X, V, p_best, l_best, g_best, w, c1, c2, PSOversion, maxd, use_bounds, centroid, a, b)
    u    = Uniform(0, 1)
    Vnew = deepcopy(V)
    Xnew = deepcopy(X)
    # update the rows independently
    msize = size(X)
    nrow  = msize[1]
    K     = msize[2]
    alpha = fill(1, K)
    d     = Dirichlet(alpha)


    # pick the correct comparitor for the chosen communication topology
    if PSOversion == "gbest"
       groupbest = g_best
    elseif PSOversion == "lbest"
       groupbest = l_best
    end
    rset = [1:1:nrow;]
    pset = shuffle(rset)
    for j in pset
        ## update velocity
        # inertia
        inertia   = Simplex.simplex_multiply(V[j,:], w)
        # cognitive
        ct1       = Simplex.simplex_multiply(X[j,:], -1.0)
        ct2       = Simplex.simplex_add(p_best[j,:], ct1)
        ctscaler  = c1*rand(u)
        cognitive = Simplex.simplex_multiply(ct2, ctscaler)
        # social
        st1       = Simplex.simplex_multiply(X[j,:], -1.0)
        st2       = Simplex.simplex_add(groupbest[j,:], st1)
        sscaler   = c2*rand(u)
        social    = Simplex.simplex_multiply(st2, sscaler)
        # velocity update
        vt        = Simplex.simplex_add(inertia, Simplex.simplex_add(cognitive, social))
        nvt       = Simplex.simplex_norm(vt)
        if nvt > maxd
            vt = Simplex.simplex_scale(vt, maxd)
        end
        Vnew[j,:] = vt
        if ~use_bounds
            Xnew[j,:] = Simplex.simplex_add(Xnew[j,:], Vnew[j,:])
        else
            Xnt       = Simplex.simplex_add(Xnew[j,:], Vnew[j,:])
            Xnew[j,:] = Geom.hyper_polyhedron_confine(transpose(Xnt), centroid, a, b)
        end
    end
    return Xnew, Vnew
end

## update local update_pg_bests - done for particle i
function update_l_best(p_best, f_pbest, l_best, f_lbest, lbest_index, neighbors)
    # This one is a bit tricky, so here are the arugment definitions
    # NOTE:
    #       1. if input is vector or array, it is the full array (all particles)
    #          at iteration time t
    #
    #  p_best  := array of p_best locations across all particles at time t
    #  f_pbest := vector of fitness of all p_best's at time t
    #  l_best  := the local neighborhood best location for particle i (single location)
    #  f_lbest := fitness at l_best (single value)
    #  lbest_index := single value indicating the index of this particles
    #                 best neighbor in its neighbor set
    #  neighbors := particle i's vector of neighbor indices

    #  compute which neighbor currently has the best p_best location
    best_neighb_index   = argmin(f_pbest[neighbors])

    if f_pbest[neighbors][best_neighb_index] < f_lbest
        # which neighbor of particle i has the best position
        lbest_index  = neighbors[best_neighb_index]
        # what is the position of the best nerighbor
        l_best       = p_best[:, :, lbest_index]
        # what is the fitness at the best neighborhood position
        f_lbest      = f_pbest[lbest_index]
    end
    return l_best, f_lbest, lbest_index
end

function update_pg_bests(x, f,  p_best, f_pbest, g_best, f_gbest)
    # knowledge check
    if f < f_pbest
        f_pbest = f
        p_best = x
        if f_pbest < f_gbest
            f_gbest = f_pbest
            g_best  = p_best
        end
    end
    return p_best, f_pbest, g_best, f_gbest
end

end # module


