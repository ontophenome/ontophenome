function m_cResult = DLP(best_Params, Y0_TST, phen_idxs, genes_idxs, n_genes, tst_idx, phen_groups, opt, L1)

    Y0_TST = Normal_M_modified(Y0_TST);

    beta = best_Params(1);
    gamma = best_Params(2);  
    
    
    fprintf('(Params) [DLP-TEST]:: beta=%4f gamma=%4f\n', beta, gamma);
    
    %- regulization parameters   
    DLPY_opt = DLP_Params(opt, beta, gamma);
    
    L2 = eye(size(phen_groups,1),size(phen_groups,1)) - Normal_M_modified(phen_groups);
    
    Y = Y0_TST;
    
    W_sol = myfunc_lbfgsb_DLP(Y', Y0_TST', DLPY_opt.regul_tr1*L1, DLPY_opt.regul_tr2*L2, DLPY_opt);
    Y = W_sol';
        
    m_cResult.DLP_Yhat =  Y;
    
    %- Evaluation for Single-Task GL 
    m_vTotRanks = DLP_Eval(Y0_TST, Y, phen_idxs, genes_idxs, n_genes, tst_idx);
    m_cResult.DLP_RankAvg = mean(m_vTotRanks);
    m_cResult.DLP_Eval = m_vTotRanks;

end

