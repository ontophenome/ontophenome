function m_cResult = DLP_CV(Y0_CV, mm_Params, phen_idxs, genes_idxs, n_genes, cvtst_idx, phen_groups, opt, L1)
% cross-validation will find the best beta

    Y0_CV = Normal_M_modified(Y0_CV);

    m_vEvalRank = zeros(size(mm_Params,1),1);
    mvAllTotRanks = cell(size(mm_Params,1),1);
    L2 = eye(size(phen_groups,1),size(phen_groups,1)) - Normal_M_modified(phen_groups);
    
    parallelPoolStart();
    parfor mn_subiter = 1:size(mm_Params,1),
        
        %- regulization parameters
        beta = mm_Params(mn_subiter,1);  
        gamma = mm_Params(mn_subiter,2);  
        
        DLP_opt = DLP_Params(opt, beta, gamma);
        fprintf('(%2d) [DLP-CV]:: beta=%4f gamma=%4f\n', mn_subiter, beta, gamma);

        Y = Y0_CV;        
        W_sol = myfunc_lbfgsb_DLP(Y', Y0_CV', DLP_opt.regul_tr1*L1, DLP_opt.regul_tr2*L2, DLP_opt);
        Y = W_sol';
        
        %- CV Evaluation for Single-Task GL
        m_vTotRanks = DLP_Eval(Y0_CV, Y, phen_idxs, genes_idxs, n_genes, cvtst_idx, phen_groups);
        m_vEvalRank(mn_subiter) = sum(m_vTotRanks<100);
        mvAllTotRanks{mn_subiter} = m_vTotRanks;
    end
    try
        parallelPoolClose();
    catch
        try
            parallelPoolClose();
        catch 
        end
    end
    m_cResult.DLP_CVresult = m_vEvalRank;

    [~, mn_BestParam] = max(m_vEvalRank, [], 1);
    m_cResult.DLP_BParams = mm_Params(mn_BestParam,:);
    m_cResult.DLP_TotRanks = mvAllTotRanks;
    
end

