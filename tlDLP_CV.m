function m_cResult = tlDLP_CV(X0, Y0_CV, mm_Params, m, S, m_copt, L, m_nMaxturns, G_gf, G_ph, m_vidxR, m_vIdxC, mv_CVTstIDX, m_vGeneIDXE, Y_hat)

    X0 = Normal_M_modified(X0);
    Y0 = Y_hat;
    
    %- cross-validation
    m_vEvalRank = zeros(size(mm_Params,1),1);
    mvAllTotRanks = cell(size(mm_Params,1),1);
    
    parallelPoolStart();
    parfor mn_subiter = 1:size(mm_Params,1),
        % t = 0;
        X = X0;        
        Y = Y0_CV;
        
        %- regulization parameters
        beta_X = mm_Params(mn_subiter,1);
        gamma_X = mm_Params(mn_subiter,2);
        beta_Y = mm_Params(mn_subiter,3);
        gamma_Y = mm_Params(mn_subiter,4);
        zeta = mm_Params(mn_subiter,5);
        
        eta = 0.0;
        mm_CholL = chol(eye(m,m) - eta*S, 'upper');
        mm_invL = mm_CholL\(mm_CholL'\eye(m,m));        
        Z0 = (mm_invL*X0')'*(mm_invL*Y0');
        
        % for X
        XCV_opt = m_copt;

        XCV_opt.gamma = zeta;
        XCV_opt.regul_tr1 = beta_X;
        XCV_opt.regul_tr2 = gamma_X;
        XCV_opt.regul_glasso = 0;

        X_L1 = sparse(XCV_opt.regul_tr1*L);
        X_L2 = sparse(XCV_opt.regul_tr2*(eye(size(G_gf,1),size(G_gf,1)) - Normal_M_modified(G_gf)));
        
        % for Y                
        YCV_opt = m_copt;
        
        YCV_opt.gamma = zeta; 
        YCV_opt.regul_tr1 = beta_Y;
        YCV_opt.regul_tr2 = gamma_Y;
        YCV_opt.regul_glasso = 0;
        
        Y_L1 = sparse(YCV_opt.regul_tr1*L);                
        Y_L2 = sparse(YCV_opt.regul_tr2*(eye(size(G_ph,1),size(G_ph,1)) - Normal_M_modified(G_ph)));
        
    
        fprintf('(%3d) [tlDLP-CV]:: betaX=%4f, gammaX=%4f, betaY=%4f, gammaY=%4f, zeta=%4f \n', ...
                                     mn_subiter,beta_X,gamma_X,beta_Y,gamma_Y,zeta);

        first_obj_val_X = -1;
        old_obj_val_X = -1;
        first_obj_val_Y = -1;
        old_obj_val_Y = -1;
        should_continue_X = true;
        should_continue_Y = true;
        for m_nt = 1:m_nMaxturns,
            
            %- Update X:
            if should_continue_X
                [X_sol, m_vobjs] = myfunc_lbfgsb_tlDLP(X', X0', Z0, Y', X_L1, X_L2, XCV_opt);
                X = X_sol';

                if first_obj_val_X == -1
                    first_obj_val_X = m_vobjs(1);
                    fprintf('first X value: %4f \n', first_obj_val_X);
                else
                    if length(m_vobjs) > 0
                        diff = ((old_obj_val_X - m_vobjs(1))/first_obj_val_X);
                        fprintf('X value: old %4f new %4f diff %4f \n', old_obj_val_X, m_vobjs(1), diff);
                        if diff <= 0.005
                            should_continue_X = false;
                        end
                    end
                end
                if length(m_vobjs) > 0
                    old_obj_val_X = m_vobjs(1);
                end
            end
            
            %- Update Y: 
            if should_continue_Y
                [Y_sol, m_vobjs] = myfunc_lbfgsb_tlDLP(Y', Y0', Z0', X', Y_L1, Y_L2, YCV_opt);
                Y = Y_sol';

                if first_obj_val_Y == -1
                    first_obj_val_Y = m_vobjs(1);
                    fprintf('first Y value: %4f \n', first_obj_val_Y);
                else
                    if length(m_vobjs) > 0
                        diff = ((old_obj_val_Y - m_vobjs(1))/first_obj_val_Y);
                        fprintf('Y value: old %4f new %4f diff %4f \n', old_obj_val_Y, m_vobjs(1), diff);
                        if diff <= 0.005
                            should_continue_Y = false;
                        end
                    end
                end
                if length(m_vobjs) > 0
                    old_obj_val_Y = m_vobjs(1);
                end
            end
            
            if ~should_continue_X && ~should_continue_Y
                break;
            end
            
            
        end

        %- CV Evaluation for Multi-Task GL
        m_vNewIDXR = m_vidxR(mv_CVTstIDX);
        m_vNewIDXC = m_vIdxC(mv_CVTstIDX);
        
        mv_uniqIRs = unique(m_vNewIDXR);   

        m_vTotRanks = [];
        for mn_i = 1:length(mv_uniqIRs),
            mn_pos = mv_uniqIRs(mn_i);
            
            if sum(G_ph(:,mn_pos)) ~= 1 % only leaf
                continue;
            end
            
            m_vTrueIDX = find(Y0_CV(mn_pos,:));
            m_vCandIDX = m_vNewIDXC(m_vNewIDXR==mn_pos);  

            m_vCurGeneIDXE = m_vGeneIDXE;
            m_vCurGeneIDXE([m_vTrueIDX, m_vCandIDX']) = [];
            
            m_vRank = zeros(length(m_vCandIDX), 1);
            for mn_j = 1:length(m_vCandIDX),
                m_vEstVals = full(Y(mn_pos, [m_vCandIDX(mn_j), m_vCurGeneIDXE]));      
                [~, mv_idxs] = sort(m_vEstVals, 2, 'descend');
            
                m_vRank(mn_j) = find(mv_idxs == 1) + sum(m_vEstVals(1)==m_vEstVals) - 1;
            end

            m_vTotRanks = [m_vTotRanks; m_vRank]; %#ok<AGROW>           
        end
        
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
    
    m_cResult.tlDLP_CVresult = m_vEvalRank;
    
    %- Test step
    [~, mn_BestParam] = max(m_vEvalRank, [], 1);    
    m_cResult.tlDLP_BParams = mm_Params(mn_BestParam, :);  
    m_cResult.tlDLP_TotRanks = mvAllTotRanks;

end

