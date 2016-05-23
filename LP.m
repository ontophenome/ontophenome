function m_cResult = LP(etas, Y0_CV, Y0_TST, phen_idxs, genes_idxs, n_genes, cvtst_idx, tst_idx, m, S)
    
    m_vEvalRank = zeros(length(etas),1);

    for mn_subiter = 1:length(etas),                    
        %- regulization parameters
        eta = etas(mn_subiter);  
        
        fprintf('(%3d) [LP-CV]:: eta=%4f \n', mn_subiter, eta);
        
        mm_CholL = chol(eye(m,m) - eta*S, 'upper');
        mm_invL = mm_CholL\(mm_CholL'\eye(m,m));        
        Y = (mm_invL*Normal_M_modified(Y0_CV)')';
                
        %- CV evaluation for Single-Task with Lasso
        m_vTotRanks = LP_Eval(Y0_CV, Y, phen_idxs, genes_idxs, n_genes, cvtst_idx);
        m_vEvalRank(mn_subiter) = sum(m_vTotRanks<100);
    end
    m_cResult.LP_CVresult = m_vEvalRank;
       
    %% - Test step
    %- regulization parameters
    [~, mn_BestParam] = max(m_vEvalRank, [], 1);    
    
    eta = etas(mn_BestParam);               
    m_cResult.LP_BParams = eta;   
    
    fprintf('(Params) [LP-TEST]:: beta=%4f \n', eta);
    
    mm_CholL = chol(eye(m,m) - eta*S, 'upper');
    mm_invL = mm_CholL\(mm_CholL'\eye(m,m));
    Y = (mm_invL*Normal_M_modified(Y0_TST)')';

    m_cResult.LP_Yhat =  Y;
    
    % Evaluation 
    m_vTotRanks = LP_Eval(Y0_TST, Y, phen_idxs, genes_idxs, n_genes, tst_idx);
    
    m_cResult.LP_RankAvg = mean(m_vTotRanks);
    m_cResult.LP_Eval = m_vTotRanks;

end

