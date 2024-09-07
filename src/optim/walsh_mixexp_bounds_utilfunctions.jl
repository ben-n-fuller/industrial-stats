
## developt truncated dirchlet sampling and polyhedral confinement
#print(pwd())
#cd("C:/Users/Stephen Walsh/Desktop/julia_mixture_devel")

# required packages
#using Distributions, LinearAlgebra, Random, SpecialFunctions, RCall, StatsPlots, Combinatorics


## Functions for working with lower and upper consrtaints inside the simplex

function tBetaQuantMarg(k, a, b,  a_plus, b_plus, gam, gam_plus, u)
    # sim from beta marginal to Dirichlet

    # marg param quants
    xi_nm1   = max(a[k], 1 - b_plus + b[k])
    eta_nm1  = min(b[k], 1 - a_plus + a[k])

    # place holder
    gam1_t   = gam[k]
    # sim from the Beta parts
    Beta_t   = Beta(gam1_t, gam_plus - gam1_t)
    t1       = u[k] * (cdf(Beta_t, eta_nm1) - cdf(Beta_t, xi_nm1))
    t2       = cdf(Beta_t, xi_nm1) + t1
    qt       = quantile(Beta_t, t2)
    return qt
end

function tBetaQuantCond(k, x, a, b,  a_plus, b_plus, gam,  gam_plus, u)
    # sim from beta conditional to Dirichlet
    n        = length(a)
    t0       = (1 - sum( x[( k + 1):( n - 1)] ))
    t1       = a[k] / t0
    t2       = 1 - (b_plus - sum( b[ k:( n - 1) ] ) ) / t0
    xi_k     = max(t1, t2)

    t3       = b[k] / t0
    t4       = 1 - (a_plus - sum( a[ k:( n - 1) ] )) / t0
    eta_k    = min(t3, t4)

    t5       = (1 - sum( x[( k + 1):( n - 1)] ))

    gamk_t   = sum( gam[ k:(n - 1 ) ] )
    # sim from the Beta parts
    Beta_t   = Beta(gam[k], gam_plus - gamk_t )
    t6_1     = u[k] * (cdf(Beta_t, eta_k) - cdf(Beta_t, xi_k))
    t6_2     = cdf(Beta_t, xi_k) + t6_1
    t6       = quantile(Beta_t, t6_2)
    qt       = t5*t6
    return(qt)
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


function rTDirichlet(a, b; N = 1, gamma = undef)
    # checks
    K        = length(a)
    K2       = length(b)
    if ~(K == K2)
        error("(rTDirichlet): lower and upper bound vectors are not the same length!")
    end
    a_plus   = sum(a)
    b_plus   = sum(b)
    # set Dirichlet parameters to uniform if not specified
    if gamma == undef
        gamma = fill(1, K)
    end
    gam_plus = sum(gamma)
    ## allocate space for data
    data     = Matrix{Float64}(undef,N,K)
    # function to simulate from a beta marginal to Dirichlet
    for i in 1:N
        U         = Uniform(0,1)
        u         = rand(U, K-1)
        x         = Vector{Float64}(undef, K-1)
        x[K-1]    = tBetaQuantMarg(K-1, a, b,  a_plus, b_plus, gamma, gam_plus, u)
        for k in (K-2):-1:1
            x[k]  = tBetaQuantCond(k, x, a, b,  a_plus, b_plus, gamma,  gam_plus, u)
        end
        x         = vcat(x, 1-sum(x))
        data[i,:] = x
    end
    checkRanges(data, a, b)
    return data
end

# these functions implement the gibbs sampler on the polytope

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

function tDirGibbs(N, a, b, gamma, centroid)

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







function checkPolyhedronConsistency(a, b)
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

## cornell ch3 pp 120
#  crosier 1986


function  countVEF(a, b)
    #  count vertices edges and faces from
    #     A primer on Mixtures - Cornell, 2011, Sewc 3.8 pp 119
    # number components
    K  = length(a)
    # number of d-dimensional boundarys is,
    # for d = 0, 1, 2, ..., K-2
    ds = collect(0:1:(K-2))
    nd = length(ds)
    # this is the answer of this function
    # ie its the number of
    # d = 0: vertex
    # d = 1: edge
    # d = 2: face
    # d = 3: hyperface etc.
    # ranges
    R  = round.(b - a, digits = 6)
    RL = round.(1 - sum(a), digits = 6)
    RU = round.(sum(b) - 1, digits = 6)
    Rp = min(RL, RU)

    #allocate abstract array for L, E, G vector
    LEG = Array{Any}(undef, K)
    rs = 1:1:K

    for r in rs
        colt = collect(combinations(R, r))
        cols = sum.(colt)
        Lr   = sum(cols .< Rp)
        Er   = sum(cols .== Rp)
        Gr   = sum(cols .> Rp)
        st   = Lr + Er + Gr
        et   = binomial(K, r)
        if ~(et == st)
            println("r = ", r)
            error("(countVEF) choose(K,r) != L(r) + E(r) + G(r)")
        end
        LEG[r] = [Lr Er Gr]
    end
    Nd     = NdFunc(LEG)
    dim    = 0:1:(length(Nd)-1)
    names  = ["vertex", "edge", "face", fill("hyperface", 20)]
    result = DataFrame(dimension = dim,
                        boundary_type  = names[1:length(Nd)],
                        num_boundaries = Nd)
    #= println("----------------------------------------------------------|")
    println("  PROPERTIES OF THIS POLYHEDRON:                          |")
    println("                                                          |")
    println(result)
    println("                                                          |")
    println("                                                          |")
    println("----------------------------------------------------------|")
    println("")
    println("")
    =#
    return result
end

function NdFunc(LEG)
    # given a LEG compute Nd
    q   = length(LEG)
    ds  = collect(0:1:(q-2))
    Nd  = Vector{Int64}(undef, length(ds))
    ind = 0
    for d in ds
        ind = ind + 1
        if d == 0
            t1 = q
            t2 = 0
            for r in 1:1:q
                LEGt = LEG[r]
                Lr   = LEGt[1]
                Er   = LEGt[2]
                t2   = t2 + Lr * (q - 2*r) - Er * (r - 1)
            end
            Nd[ind] = t1 + t2
        elseif d > 0
            t1 = binomial(q, q - d - 1)
            t2 = 0
            for r in 1:1:(q - d - 1)
                LEGt = LEG[r]
                Lr   = LEGt[1]
                Er   = LEGt[2]
                t2   = t2 + Lr*binomial(q - r, q - r - d - 1)
            end
            t3 = 0
            for r in (d + 1):1:q
                LEGt = LEG[r]
                Lr   = LEGt[1]
                Er   = LEGt[2]
                t3   = t3 + (Lr + Er)*binomial(r, r - d - 1)
            end
            Nd[ind] = t1 + t2 - t3
        end

    end
    return Nd
end

# function to compute sqare distance matrix

function simplexVecOuterDmat(V)
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
            D[i, j] = simplexVecDist(V[i,:], V[j,:])
        end
    end
    return D
end


function  computeCentroid(a,b)
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
function hyperPolyhedronConfine(badmix, centroid, a, b)
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

#=
Li = [0.40 0.10 0.05 0.05]
Ui = [0.80 0.50 0.30 0.30]
countVEF(Li, Ui)
centroid = computeCentroid(Li,Ui)
foo = hyperPolyhedronConfine([1 0 0 0 ], computeCentroid(Li,Ui), Li, Ui)

badmix = [1 0 0 0 ]
a = Li
b = Ui

vl = any(badmix .<= a)
vu = any(badmix .>= b)
violated = vl | vu


K       = length(centroid)
# Initiate new mixture point
newmix  = transpose(Vector{Float64}(undef, K))
# ranges
R       = b - a
L_delta = (a - badmix) ./ R
U_delta = (badmix - b) ./ R
M_delta = Vector{Float64}(undef, K)
UorB    = Vector{Float64}(undef, K)
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
    betat = (b[ind_max] - centroid[ind_max])/(badmix[ind_max] - centroid[ind_max])
    newmix[ind_max] = b[ind_max]
elseif UorB[ind_max] == -1
    betat = (a[ind_max] - centroid[ind_max])/(badmix[ind_max] - centroid[ind_max])
    newmix[ind_max] = a[ind_max]
end
newmix[ind_notmax] .= (1-betat) .* badmix[ind_notmax] .+ (betat) .* centroid[ind_notmax]

vl2 = any(newmix .<= a)
vu2 = any(newmix .>= b)
violated2 = vl | vu

=#


# Enter upper and lower component bounds
#UB = [.51 .52 .58]
#LB = [.18 .14 .11]
#cent = computeCentroid(LB,UB)

# Enter mixture outside of design space
#badmix = [.9 .05 .05]
#badmix = [.1 .7 .2]
#badmix = [0 .5 .5]
#badmix = [.3 .4 .3]
#hyperPolyhedronConfine(badmix, [0.3383    0.3233    0.3383], LB, UB)

## MC approach to calculating the centroid
# bounds

#=
function computeCentroid(a,b; N = 100000)
    iter     = 2
    K        = length(a)
    rel_tol  = 1.0e-8
    Xt       = rTDirichlet(a, b, N = N)
    mean_t   = mean(Xt, dims = 1)
    crit     = fill(typemax(Float64))
    runwhile = true

    println("=========================================")
    println(" ... computing polyhedron centroid ...  |")

    while runwhile
        iter      = iter + 1
        mean_prev = mean_t
        Xt        = vcat(Xt, rTDirichlet(a, b, N = 1))
        mean_t    = mean(Xt, dims = 1)
        crit      = abs.(mean_t - mean_prev)
        runwhile  = any(crit .> rel_tol)
    end

    println(" ...                                    |")
    println(" finished.                              |")
    println("=========================================")
    return mean_t, iter
end
=#


## verify!!! :D

# test count VEF
#=
Li = [0.8, 0.05, 0.02, 0.03]
Ui = [0.90, 0.15, 0.10, 0.05]
countVEF(Li, Ui)
computeCentroid(Li,Ui)

Li = [0.20 0.20 0.18]
Ui = [0.40 0.60 0.70]
countVEF(Li, Ui)
computeCentroid(Li,Ui)

Li = [0.40 0.10 0.05 0.05]
Ui = [0.80 0.50 0.30 0.30]
countVEF(Li, Ui)
computeCentroid(Li,Ui)
foo = hyperPolyhedronConfine([1 0 0 0 ], computeCentroid(Li,Ui), Li, Ui)
sum(foo)

#rTDirichlet(a,b; N = 10000, gamma = [1 1000 500])

# ex 1
#Li = [0.18, 0.14 ,0.11]
#Ui = [0.51, 0.52, 0.58]

# ex2 jobo space fill high dim
#Li = [0.1, 0, 0.1]
#Ui = [0.7, 0.8, 0.6]

# jobo page 251 ex
#Li = [0.2, 0.1, 0.1]
#Ui = [0.6, 0.6, 0.5]
#ct = [0.383 0.333 0.283]

# suffer through one triangle example
#Li = [0.2, 0.2, 0.18]
#Ui = [0.4, 0.60, 0.70]


Li = [0.3 0.3 0.3]
Ui = [1 1 1]

Li = [0.1 0.1 0.1]
Ui = [1 1 1]

Li = [0.2 0.1 0.1]
Ui = [0.6 0.6 0.5]

Li = [0.1 0 0.1]
Ui = [0.7 0.8 0.6]

Li, Ui = checkPolyhedronConsistency(Li, Ui)
cent, v = computeCentroid(Li, Ui)


D = tDirGibbs(5000, Li, Ui, [1 1 1], cent)
ct2 = mean(D, dims = 1)
println(ct2)

@rput D
@rput Li
@rput Ui
@rput cent
@rput ct2
R"""
library(ggtern)
colnames(D) <- c("x1", "x2", "x3")
colnames(cent) <- c("x1", "x2", "x3")
colnames(ct2) <- c("x1", "x2", "x3")
lines <- data.frame(x1 = c(0.5, 0, 0.5),
                    x2 = c(0.5, 0.5, 0),
                    x3 = c(0, 0.5, 0.5),
                    xend = c(1, 1, 1)/3,
                    yend = c(1, 1, 1)/3,
                    zend = c(1, 1, 1)/3)
ggtern(data = as.data.frame(D), aes(x1, x2, x3)) +
    geom_segment(data = lines, aes(x = x1, y = x2, z = x3,
                                  xend = xend,yend = yend, zend = zend),
                color = 'grey', size = 0.2) +
    geom_point(size = 0.7, color = "goldenrod1") + theme_bw() + theme_nomask() + theme_clockwise() +
  geom_Lline(Lintercept = c(Li[1], Ui[1]), linetype = "dashed", color = "red", size = 2) +
  geom_Tline(Tintercept = c(Li[2], Ui[2]), linetype = "dashed", color = "blue", size = 2) +
  geom_Rline(Rintercept = c(Li[3], Ui[3]), linetype = "dashed", color = "forestgreen", size = 2)+ theme_rotate(degrees = -120) +
  geom_point(data = as.data.frame(cent), size = 3, color = "black") +
  geom_point(data = as.data.frame(ct2), size = 3, color = "magenta", shape = 17)
"""

K = 3
D1 = rTDirichlet(fill(0,K), fill(1,K); N = 10000, gamma = fill(1,K))
D2 = rTDirichlet(fill(0,K), fill(1,K); N = 10000, gamma = fill(5,K))
D3 = rTDirichlet(fill(0,K), fill(1,K); N = 5000, gamma = fill(10,K))
D4 = rTDirichlet(fill(0,K), fill(1,K); N = 700, gamma = fill(20,K))
D5 = rTDirichlet(fill(0,K), fill(1,K); N = 1250, gamma = fill(40,K))
D6 = rTDirichlet(fill(0,K), fill(1,K); N = 3000, gamma = fill(100,K))
D7 = rTDirichlet(fill(0,K), fill(1,K); N = 1000, gamma = fill(300,K))

@rput D1 D2 D3 D4 D5 D6 D7
R"""
library(ggtern)
colnames(D1) <- c("x1", "x2", "x3")
colnames(D2) <- c("x1", "x2", "x3")
colnames(D3) <- c("x1", "x2", "x3")
colnames(D4) <- c("x1", "x2", "x3")
colnames(D5) <- c("x1", "x2", "x3")
colnames(D6) <- c("x1", "x2", "x3")
colnames(D7) <- c("x1", "x2", "x3")
lines <- data.frame(x1 = c(0.5, 0, 0.5),
                    x2 = c(0.5, 0.5, 0),
                    x3 = c(0, 0.5, 0.5),
                    xend = c(1, 1, 1)/3,
                    yend = c(1, 1, 1)/3,
                    zend = c(1, 1, 1)/3)
ggtern(data = as.data.frame(D1), aes(x1, x2, x3)) +
    geom_segment(data = lines, aes(x = x1, y = x2, z = x3,
                                  xend = xend,yend = yend, zend = zend),
                color = 'grey', size = 0.2) +
    theme_darker() + theme_nomask() + theme_clockwise() +
    theme_rotate(degrees = -120) +
    geom_point(size = 1, color = "salmon") +
    geom_point(data = as.data.frame(D2), size = 1, color = "white") +
    geom_point(data = as.data.frame(D3), size = 1, color = "blue") +
    geom_point(data = as.data.frame(D4), size = 1, color = "green") +
    geom_point(data = as.data.frame(D5), size = 1, color = "royalblue1") +
    geom_point(data = as.data.frame(D6), size = 1, color = "grey30") +
    geom_point(data = as.data.frame(D7), size = 1, color = "black")
"""


K = 3
D1 = rTDirichlet(fill(0,K), fill(1,K); N = 10000)


@rput D1 D2 D3 D4 D5 D6 D7
R"""
library(ggtern)
colnames(D1) <- c("x1", "x2", "x3")

lines <- data.frame(x1 = c(0.5, 0, 0.5),
                    x2 = c(0.5, 0.5, 0),
                    x3 = c(0, 0.5, 0.5),
                    xend = c(1, 1, 1)/3,
                    yend = c(1, 1, 1)/3,
                    zend = c(1, 1, 1)/3)
ggtern(data = as.data.frame(D1), aes(x1, x2, x3)) +
    geom_segment(data = lines, aes(x = x1, y = x2, z = x3,
                                  xend = xend,yend = yend, zend = zend),
                color = 'grey', size = 0.2) +
    theme_darker() + theme_nomask() + theme_clockwise() +
    theme_rotate(degrees = -120) +
    geom_point(size = 0.5, color = "blue")
"""
=#

## check confinment
#=
Li = [0.40 0.10 0.05 0.05]
Ui = [0.80 0.50 0.30 0.30]
Li, Ui = checkPolyhedronConsistency(Li, Ui)
D      = rTDirichlet(Li, Ui; N = 50000, gamma = [1 1 1 1])
println(round.(minimum(D,dims =1), digits = 3))
println(round.(maximum(D,dims =1), digits = 3))
ct2    = mean(D, dims = 1)
cent   = computeCentroid(Li, Ui)

@rput D
@rput Li
@rput Ui
@rput cent
@rput ct2
R"""
library(ggtern)
colnames(D)    <- c("x1", "x2", "x3", "x4")
colnames(cent) <- c("x1", "x2", "x3", "x4")
colnames(ct2)  <- c("x1", "x2", "x3", "x4")
lines <- data.frame(x1   = c(0.5, 0, 0.5, 0),
                    x2   = c(0.5, 0.5, 0, 0),
                    x3   = c(0, 0.5, 0.5, 0),
                    x4   = c(0, 0, 0.5, 0.5),
                    xend = rep(1,4)/4,
                    yend = rep(1,4)/4,
                    zend = rep(1,4)/4,
                    qend = rep(1,4)/4)

p1 <-   ggtern(data = as.data.frame(D), aes(x1, x2, x3)) +
        geom_segment(data = lines, aes(x = x1, y = x2, z = x3,
                     xend = xend,yend = yend, zend = zend), color = 'grey', size = 0.2) +
        geom_point(size = 0.7, color = "skyblue") + theme_bw() + theme_nomask() + theme_clockwise() +
        geom_Lline(Lintercept = c(Li[1], Ui[1]), linetype = "dashed", color = "red", size = 1.3) +
        geom_Tline(Tintercept = c(Li[2], Ui[2]), linetype = "dashed", color = "blue", size = 1.3) +
        geom_Rline(Rintercept = c(Li[3], Ui[3]), linetype = "dashed", color = "forestgreen", size = 1.3)+ theme_rotate(degrees = -120) +
        geom_point(data = as.data.frame(cent), size = 3, color = "black") +
        geom_point(data = as.data.frame(ct2), size = 3, color = "magenta", shape = 17)

p2 <-   ggtern(data = as.data.frame(D), aes(x1, x2, x4)) +
        geom_segment(data = lines, aes(x = x1, y = x2, z = x4,
                     xend = xend, yend = yend, zend = qend), color = 'grey', size = 0.2) +
        geom_point(size = 0.7, color = "skyblue") + theme_bw() + theme_nomask() + theme_clockwise() +
        geom_Lline(Lintercept = c(Li[1], Ui[1]), linetype = "dashed", color = "red", size = 1.3) +
        geom_Tline(Tintercept = c(Li[2], Ui[2]), linetype = "dashed", color = "blue", size = 1.3) +
        geom_Rline(Rintercept = c(Li[4], Ui[4]), linetype = "dashed", color = "orange", size = 1.3)+ theme_rotate(degrees = -120) +
        geom_point(data = as.data.frame(cent), size = 3, color = "black") +
        geom_point(data = as.data.frame(ct2), size = 3, color = "magenta", shape = 17)

p3 <-   ggtern(data = as.data.frame(D), aes(x2, x3, x4)) +
        geom_segment(data = lines, aes(x = x2, y = x3, z = x4,
                             xend = yend, yend = zend, zend = qend), color = 'grey', size = 0.2) +
        geom_point(size = 0.7, color = "skyblue") + theme_bw() + theme_nomask() + theme_clockwise() +
        geom_Lline(Lintercept = c(Li[2], Ui[2]), linetype = "dashed", color = "blue", size = 1.3) +
        geom_Tline(Tintercept = c(Li[3], Ui[3]), linetype = "dashed", color = "forestgreen", size = 1.3) +
        geom_Rline(Rintercept = c(Li[4], Ui[4]), linetype = "dashed", color = "orange", size = 1.3) + theme_rotate(degrees = -120) +
        geom_point(data = as.data.frame(cent), size = 3, color = "black") +
        geom_point(data = as.data.frame(ct2), size = 3, color = "magenta", shape = 17)

grid.arrange(p1, p2, p3, ncol = 3)
"""
=#
