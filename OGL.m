function m_cResult = OGL(best_Params, Y0_TST, phen_idxs, genes_idxs, n_genes, tst_idx, phen_groups, phen_groups_n_ass, opt, L, d_ph)

    Y0_TST = Normal_M_modified(Y0_TST);

    beta = best_Params(1);
    gamma = best_Params(2);  
    
    
    fprintf('(Params) [OGL-TEST]:: beta=%4f gamma=%4f\n', beta, gamma);
    
    %- regulization parameters   
    OGLY_opt = OGL_Params(opt, beta, gamma);
    
    Y = Y0_TST;
    
    W_sol = myfunc_lbfgsb_OGL(Y', Y0_TST', OGLY_opt.regul_tr*L, phen_groups, phen_groups_n_ass, OGLY_opt);
    Y = W_sol';
        
    m_cResult.OGL_Yhat =  Y;
    
    %- Evaluation for Single-Task GL 
    m_vTotRanks = OGL_Eval(Y0_TST, Y, phen_idxs, genes_idxs, n_genes, tst_idx);
    m_cResult.OGL_RankAvg = mean(m_vTotRanks);
    m_cResult.OGL_Eval = m_vTotRanks;

end

