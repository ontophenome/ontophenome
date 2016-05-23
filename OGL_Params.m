function OGLY_opt = OGL_Params(opt, beta, gamma)
    OGLY_opt = opt;
    OGLY_opt.maxIts = 150;
    OGLY_opt.maxTotalIts = 500;
    OGLY_opt.regul_tr = beta;
    OGLY_opt.regul_glasso = gamma;
    OGLY_opt.mu = 0.1;
end

