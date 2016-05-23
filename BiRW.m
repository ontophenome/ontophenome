function m_cResult = BiRW(mm_Params, Y0_CV, Y0_TST, P, phen_idxs, genes_idxs, n_genes, tst_idx, S)
    
    norm_Y0_CV = Normal_M_modified(Y0_CV);
    norm_Y0_TST = Normal_M_modified(Y0_TST);

    m_vEvalRank = zeros(size(mm_Params,1),1);
    G = S;
    
    A = norm_Y0_CV ./ repmat(sum(norm_Y0_CV),size(norm_Y0_CV,1),1);
    A(isnan(A))=0;
    
    for mn_subiter = 1:size(mm_Params,1)
        %- regulization parameters
        alpha = mm_Params(mn_subiter,1);  
        l=mm_Params(mn_subiter,2);
        r=l; 
        
        fprintf('(%3d) [BiRW-CV]:: alpha=%4f, l=%4f, r=%4f \n', mn_subiter, alpha, l, r);
        
        R = A;
        for i=1:max([l,r])
            
            if i <= l
                R_l = alpha*sparse(full(P)*full(R)) + (1-alpha)*A;
            end
            if i <= r
                R_r = alpha*sparse(full(R)*full(G)) + (1-alpha)*A;
            end
            
            d_l = (i <= l);
            d_r = (i <= r);
            R_new = (d_l*R_l+d_r*R_r)/(d_l+d_r);
            
            diff = norm(full(R_new-R),'inf');
            R = R_new;
            fprintf('(%3d) [BiRW-CV]:: alpha=%4f norm=%4f \n', i, alpha, diff);
            
        end
        Y = R;
                
        %- CV evaluation for BiRW
        m_vTotRanks = BiRW_Eval(Y0_CV, Y, phen_idxs, genes_idxs, n_genes, tst_idx);     
        m_vEvalRank(mn_subiter) = sum(m_vTotRanks<100);
    end
    m_cResult.BiRW_CVresult = m_vEvalRank;
       
    %% - Test step 
    %- regulization parameters
    [mv_maxvals, mn_BestParam] = max(m_vEvalRank, [], 1);    
    
    alpha = mm_Params(mn_BestParam,1);
    l = mm_Params(mn_BestParam,2);
    r = l; 
    m_cResult.BiRW_BParams = [alpha,l,r];   

    fprintf('(Params) [BiRW-TEST]:: alpha=%4f, l=%4f, r=%4f \n', alpha, l, r);
    
    A = norm_Y0_TST ./ repmat(sum(norm_Y0_TST),size(norm_Y0_TST,1),1);
    A(isnan(A))=0;
    R = A;
    for i=1:max([l,r])

        if i <= l
            R_l = alpha*sparse(full(P)*full(R)) + (1-alpha)*A;
        end
        if i <= r
            R_r = alpha*sparse(full(R)*full(G)) + (1-alpha)*A;
        end

        d_l = (i <= l);
        d_r = (i <= r);
        R_new = (d_l*R_l+d_r*R_r)/(d_l+d_r);

        diff = norm(full(R_new-R),'inf');
        R = R_new;
        if diff < 1e-4
            break
        end
        fprintf('(%3d) [BiRW-TEST]:: alpha=%4f norm=%4f \n', i, alpha, diff);
        
    end
    Y = R;

    m_cResult.BiRW_Yhat =  Y;
    
    % Evaluation 
    m_vTotRanks = BiRW_Eval(Y0_TST, Y, phen_idxs, genes_idxs, n_genes, tst_idx);
    
    m_cResult.BiRW_RankAvg = mean(m_vTotRanks);
    m_cResult.BiRW_Eval = m_vTotRanks;

end
