module OptimalityCriteria

export d_criterion, d_criterion_2, i_criterion, g_criterion

using SpecialFunctions
using LinearAlgebra

function d_criterion(X::Array{Float64, 2})
    score = abs(det(X' * X))
    if score == 0
        return Inf
    else
        return 1 / score
    end
end

function d_criterion_2(X::Array{Float64, 2})
    score = det(X' * X)
    return -log(score)
end

function d_criterion(X::Array{Float64, 3})
    mapslices(d_criterion, X, dims=[2,3])
end

function compute_i_score(x, V, B)
    xpx = transpose(x) * x
    determinant = det(xpx)
    if determinant <= 0
        return Inf
    else
        return tr(inv(xpx) * B) / V
    end
end

function i_criterion(X::Array{Float64, 2}, model_builder::Function; model="scheffe_2")
    K = size(X, 2)
    Xm = model_builder(X)
    B = get_region_moments_matrix(K, model)
    V = 1 / gamma(K)
    compute_i_score(Xm, V, B)
end

function i_criterion(X::Array{Float64, 3}, model_builder::Function; model="scheffe_2")
    K = size(X, 3)
    Xm = model_builder(X)
    B = get_region_moments_matrix(K, model)
    V = 1 / gamma(K)
    scores = zeros(size(X, 1))
    Threads.@threads for i in 1:size(X, 1)
        scores[i] = compute_i_score(Xm[i, :, :], V, B)
    end
    scores
end

function i_criterion(model_builder::Function; model="scheffe_2")
    (X) -> i_criterion(X, model_builder, model=model)
end

function get_region_moments_matrix(K, model)
    if model == "scheffe_2" && K == 3
        B = [ 1/12  1/24  1/24  1/60  1/60  1/120 ;
              1/24  1/12  1/24  1/60  1/120 1/60  ;
              1/24  1/24  1/12  1/120 1/60  1/60  ;
              1/60  1/60  1/120 1/180 1/360 1/360 ;
              1/60  1/120 1/60  1/360 1/180 1/360 ;
              1/120 1/60  1/60  1/360 1/360 1/180  ]
    elseif model == "special_cubic" && K == 3
        B = [ 1/12  1/24  1/24  1/60   1/60   1/120  1/360  ;
              1/24  1/12  1/24  1/60   1/120  1/60   1/360  ;
              1/24  1/24  1/12  1/120  1/60   1/60   1/360  ;
              1/60  1/60  1/120 1/180  1/360  1/360  1/1260 ;
              1/60  1/120 1/60  1/360  1/180  1/360  1/1260 ;
              1/120 1/60  1/60  1/360  1/360  1/180  1/1260 ;
              1/360 1/360 1/360 1/1260 1/1260 1/1260 1/5040  ]
    elseif model == "scheffe_2" && K == 4
        B = [ 1/60  1/120 1/120 1/120 1/360  1/360  1/360  1/720  1/720  1/720  ;
              1/120 1/60  1/120 1/120 1/360  1/720  1/720  1/360  1/360  1/720  ;
              1/120 1/120 1/60  1/120 1/720  1/360  1/720  1/360  1/720  1/360  ;
              1/120 1/120 1/120 1/60  1/720  1/720  1/360  1/720  1/360  1/360  ;
              1/360 1/360 1/720 1/720 1/1260 1/2520 1/2520 1/2520 1/2520 1/5040 ;
              1/360 1/720 1/360 1/720 1/2520 1/1260 1/2520 1/2520 1/5040 1/2520 ;
              1/360 1/720 1/720 1/360 1/2520 1/2520 1/1260 1/5040 1/2520 1/2520 ;
              1/720 1/360 1/360 1/720 1/2520 1/2520 1/5040 1/1260 1/2520 1/2520 ;
              1/720 1/360 1/720 1/360 1/2520 1/5040 1/2520 1/2520 1/1260 1/2520 ;
              1/720 1/720 1/360 1/360 1/5040 1/2520 1/2520 1/2520 1/2520 1/1260  ]
    elseif model == "special_cubic" && K == 4
        B = [ 1/60   1/120  1/120  1/120  1/360   1/360   1/360   1/720   1/720   1/720   1/2520  1/2520  1/2520  1/5040 ;
              1/120  1/60   1/120  1/120  1/360   1/720   1/720   1/360   1/360   1/720   1/2520  1/2520  1/5040  1/2520 ;
              1/120  1/120  1/60   1/120  1/720   1/360   1/720   1/360   1/720   1/360   1/2520  1/5040  1/2520  1/2520 ;
              1/120  1/120  1/120  1/60   1/720   1/720   1/360   1/720   1/360   1/360   1/5040  1/2520  1/2520  1/2520  ;
              1/360  1/360  1/720  1/720  1/1260  1/2520  1/2520  1/2520  1/2520  1/5040  1/10080 1/10080 1/20160 1/20160 ;
              1/360  1/720  1/360  1/720  1/2520  1/1260  1/2520  1/2520  1/5040  1/2520  1/10080 1/20160 1/10080 1/20160 ;
              1/360  1/720  1/720  1/360  1/2520  1/2520  1/1260  1/5040  1/2520  1/2520  1/20160 1/10080 1/10080 1/20160 ;
              1/720  1/360  1/360  1/720  1/2520  1/2520  1/5040  1/1260  1/2520  1/2520  1/10080 1/20160 1/20160 1/10080 ;
              1/720  1/360  1/720  1/360  1/2520  1/5040  1/2520  1/2520  1/1260  1/2520  1/20160 1/10080 1/20160 1/10080 ;
              1/720  1/720  1/360  1/360  1/5040  1/2520  1/2520  1/2520  1/2520  1/1260  1/20160 1/20160 1/10080 1/10080 ;
              1/2520 1/2520 1/2520 1/5040 1/10080 1/10080 1/20160 1/10080 1/20160 1/20160 1/45360 1/90720 1/90720 1/90720 ;
              1/2520 1/2520 1/5040 1/2520 1/10080 1/20160 1/10080 1/20160 1/10080 1/20160 1/90720 1/45360 1/90720 1/90720 ;
              1/2520 1/5040 1/2520 1/2520 1/20160 1/10080 1/10080 1/20160 1/20160 1/10080 1/90720 1/90720 1/45360 1/90720 ;
              1/5040 1/2520 1/2520 1/2520 1/20160 1/20160 1/20160 1/10080 1/10080 1/10080 1/90720 1/90720 1/90720 1/45360  ]
    else
        error("(I_criterion) moment matrix for K=$K and model=$model not yet implemented")
    end
end

## G-criterion -----------------------------------------------------------------
function G_criterion(X; order)
    ## function eval
     K = size(X)[2]
     Xm          = genDesignMat_mix(X, order = order)
     msize       = size(Xm)
     N           = msize[1]
     p           = msize[2]
     XpX         = transpose(Xm)*Xm
 
     # we need inv(XpX), first block matrices with small determinants
     determinant = det(XpX)
 
 
     if order == 3.2
         Xpred = Xpred32
     elseif order == 3.3
         Xpred = Xpred33
     elseif order == 3.4
         Xpred = Xpred34
     elseif order == 3.41
         Xpred = Xpred341
     elseif order == 4.2
         Xpred = Xpred42
     elseif order == 4.3
         Xpred = Xpred43
     elseif order == 4.4
         Xpred = Xpred44
     else
         stop("G-grid for this number of factors and model order is not implemented!!")
     end
 
     if determinant < eps()^3
         #println(determinant)
          result = Inf #eps()^(1/2)
     else
         #H = Xpred * (XpX \ transpose(Xpred))
         #H = Xpred * (\(XpX, transpose(Xpred)))
         #D = diag(N.*H)
         #Mx = maximum(D)
         C  = cholesky(XpX, check = false)
         Z  = \(transpose(C.U), transpose(Xpred))
         T  = @. N*Z^2
         D  = sum(T, dims = 1)
         Mx = maximum(D)
         #result = 100*p/Mx
         result = Mx
 
     end
 
     return result
 end

end