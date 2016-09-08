function m_cResult = OGL_CV(Y0_CV, mm_Params, phen_idxs, genes_idxs, n_genes, cvtst_idx, phen_groups, phen_groups_n_ass, opt, L)
% cross-validation will find the best beta

    Y0_CV = Normal_M_modified(Y0_CV);

    m_vEvalRank = zeros(size(mm_Params,1),1);
    mvAllTotRanks = cell(size(mm_Params,1),1);
    
    parallelPoolStart();
    parfor mn_subiter = 1:size(mm_Params,1),
        %- regulization parameters
        beta = mm_Params(mn_subiter,1);  
        gamma = mm_Params(mn_subiter,2);  
        
        OGLY_opt = OGL_Params(opt, beta, gamma);
        fprintf('(%2d) [OGL-CV]:: beta=%4f gamma=%4f\n', mn_subiter, beta, gamma);

        Y = Y0_CV;

        W_sol = myfunc_lbfgsb_OGL(Y', Y0_CV', OGLY_opt.regul_tr*L, phen_groups, phen_groups_n_ass, OGLY_opt);
        Y = W_sol';
        
        %- CV Evaluation for Single-Task GL
        m_vTotRanks = OGL_Eval(Y0_CV, Y, phen_idxs, genes_idxs, n_genes, cvtst_idx);
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
    m_cResult.OGL_CVresult = m_vEvalRank;
    
    %- Test step
    % use beta found on the test set
    [~, mn_BestParam] = max(m_vEvalRank, [], 1);
    m_cResult.OGL_BParams = mm_Params(mn_BestParam,:);
    m_cResult.OGL_TotRanks = mvAllTotRanks;
    
end

