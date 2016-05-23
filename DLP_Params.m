function DLPY_opt = DLP_Params(opt, beta, gamma)
    DLPY_opt = opt;
    DLPY_opt.maxIts = 150;
    DLPY_opt.maxTotalIts = 500;
    DLPY_opt.regul_tr1 = beta;
    DLPY_opt.regul_tr2 = gamma;
    DLPY_opt.mu = 0.1;
end

