function genDesignMat_mix(X; order)
    msize = size(X)
    N     = msize[1]
    k     = msize[2]
    valid_orders = [3.1, 3.2, 3.3, 3.4, 3.41, 3.42, 4.1, 4.2, 4.3, 4.4]
    if ~in(order, valid_orders)
        throw(DomainError(order, "(genDesignMat_mix) permissable order/model specifications are: \n [k = 3; first order Scheffe    ]:      order = 3.1\n [k = 3; second-order Scheffe   ]:      order = 3.2\n [k = 3; special cubic          ]:      order = 3.3\n [k = 3; full cubic             ]:      order = 3.4\n [k = 3; modified cubic         ]:      order = 3.41\n [k = 3; mod cubic no 3X term   ]:      order = 3.42\n [k = 4; first order Scheffe    ]:      order = 4.1\n [k = 4; second-order Scheffe   ]:      order = 4.2\n [k = 4; special cubic          ]:      order = 4.3\n [k = 4; full cubic             ]:      order = 4.4"))
    end
    if ~in(k, floor.(valid_orders))
        throw(error("(genDesignMat_mix) ncol(X) (i.e. number of factors) is not permissable. permissable values are k = {3, 4}"))
    end
    if in(k, floor.(valid_orders[1:6])) &  ~in(order, valid_orders[1:6])
        throw(error("(genDesignMat_mix) ncol(X) must match order"))
    elseif in(k, floor.(valid_orders[7:10])) & ~in(order, valid_orders[7:10])
        throw(error("(genDesignMat_mix) ncol(X) must match order"))
    end
    if k == 3
        x1 = X[:, 1]
        x2 = X[:, 2]
        x3 = X[:, 3]
    elseif k == 4
        x1 = X[:, 1]
        x2 = X[:, 2]
        x3 = X[:, 3]
        x4 = X[:, 4]
    end
    if order == 3.1
        Xm = [x1 x2 x3]
    elseif order == 3.2
        Xm = [x1 x2 x3 (x1 .* x2) (x1 .* x3) (x2 .* x3)]
    elseif order == 3.3
        Xm = [x1 x2 x3 (x1 .* x2) (x1 .* x3) (x2 .* x3) (x1 .* x2 .* x3)]
    elseif order == 3.4
        Xm = [x1 x2 x3 (x1 .* x2) (x1 .* x3) (x2 .* x3) ((x1 .* x2) .* (x1 - x2)) ((x1 .* x3) .*(x1 - x3)) ((x2 .* x3) .* (x2 - x3)) (x1 .* x2 .* x3)]
    elseif order == 3.41
        Xm = [x1 x2 x3 ((x1 .* x2) .* (x1 - x2)) ((x1 .* x3) .*(x1 - x3)) ((x2 .* x3) .* (x2 - x3)) ]
    elseif order == 3.42
        Xm = [x1 x2 x3 (x1 .* x2) (x1 .* x3) (x2 .* x3) ((x1 .* x2) .* (x1 - x2)) ((x1 .* x3) .*(x1 - x3)) ((x2 .* x3) .* (x2 - x3)) ]
    elseif order == 4.1
        Xm = [x1 x2 x3 x4]
    elseif order == 4.2
        Xm = [x1 x2 x3 x4 (x1 .* x2) (x1 .* x3) (x1 .* x4) (x2 .* x3) (x2 .* x4) (x3 .* x4)]
    elseif order == 4.3
        Xm = [x1 x2 x3 x4 (x1 .* x2) (x1 .* x3) (x1 .* x4) (x2 .* x3) (x2 .* x4) (x3 .* x4) (x1 .* x2 .* x3) (x1 .* x2 .* x4) (x1 .* x3 .* x4) (x2 .* x3 .* x4)]
    elseif order == 4.4
        Xm = [x1 x2 x3 x4 (x1 .* x2) (x1 .* x3) (x1 .* x4) (x2 .* x3) (x2 .* x4) (x3 .* x4) ((x1 .* x2) .* (x1 - x2)) ((x1 .* x3) .* (x1 - x3)) ((x1 .* x4) .* (x1 - x4)) ((x2 .* x3) .* (x2 - x3)) ((x2 .* x4) .* (x2 - x4)) ((x3 .* x4) .* (x3 - x4)) (x1 .* x2 .* x3) (x1 .* x2 .* x4) (x1 .* x3 .* x4) (x2 .* x3 .* x4)]
    end
    return Xm
end

function I_criterion2(X; order)
    k     = size(X)[2]
    Xm    = genDesignMat_mix(X, order = order)
    msize = size(Xm)
    N     = msize[1]
    p     = msize[2]
    if order == 3.2
        B = [ 1/12  1/24  1/24  1/60  1/60  1/120 ;
              1/24  1/12  1/24  1/60  1/120 1/60  ;
              1/24  1/24  1/12  1/120 1/60  1/60  ;
              1/60  1/60  1/120 1/180 1/360 1/360 ;
              1/60  1/120 1/60  1/360 1/180 1/360 ;
              1/120 1/60  1/60  1/360 1/360 1/180  ]
    elseif order == 3.3
        B = [ 1/12  1/24  1/24  1/60   1/60   1/120  1/360  ;
              1/24  1/12  1/24  1/60   1/120  1/60   1/360  ;
              1/24  1/24  1/12  1/120  1/60   1/60   1/360  ;
              1/60  1/60  1/120 1/180  1/360  1/360  1/1260 ;
              1/60  1/120 1/60  1/360  1/180  1/360  1/1260 ;
              1/120 1/60  1/60  1/360  1/360  1/180  1/1260 ;
              1/360 1/360 1/360 1/1260 1/1260 1/1260 1/5040  ]
    elseif order == 4.2
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
    elseif order == 4.3
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
        error("(I_criterion) moment matrix for that order model has not been programmed yet")
    end
    V           = 1/gamma(k)
    XpX         = transpose(Xm)*Xm
    determinant = det(XpX)
    if determinant <0
         result =  typemax(Float64)
    else
        temp1 = XpX \ B
        result = tr(temp1) / V
    end
    return result
end