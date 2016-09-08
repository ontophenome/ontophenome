function m_cResult = tlDLP(X0, tlDLP_Best_Params, m, S, m_copt, L, m_nMaxturns, G_gf, G_ph, m_vValT, m_vidxR, m_vIdxC, mv_TstIDX, m_vGeneIDXE, mv_TrnIDX, Y_hat)

    X0 = Normal_M_modified(X0);
    Y0 = Y_hat;
    
    beta_X = tlDLP_Best_Params(1,1);
    gamma_X = tlDLP_Best_Params(1,2);
    beta_Y = tlDLP_Best_Params(1,3);
    gamma_Y = tlDLP_Best_Params(1,4);
    zeta = tlDLP_Best_Params(1,5);
    
    fprintf('(Param) [tlDLP-TEST]:: betaX=%4f, gammaX=%4f, betaY=%4f, gammaY=%4f, zeta=%4f \n', ...
                                     beta_X,gamma_X,beta_Y,gamma_Y,zeta);
    % t = 0
    X = X0;
    
    m_vNewVals = m_vValT(mv_TrnIDX);
    m_vNewIDXR = m_vidxR(mv_TrnIDX);
    m_vNewIDXC = m_vIdxC(mv_TrnIDX);
    Y0_TST = sparse(m_vNewIDXR,m_vNewIDXC,m_vNewVals,size(G_ph,1),m);
    Y = Y0_TST;
    
    eta = 0.0;
    mm_CholL = chol(eye(m,m) - eta*S, 'upper');
    mm_invL = mm_CholL\(mm_CholL'\eye(m,m));        
    Z0 = (mm_invL*X0')'*(mm_invL*Y0');
    
    % for X
    X_opt = m_copt;
    X_opt.gamma = zeta;
    X_opt.regul_tr1 = beta_X;
    X_opt.regul_tr2 = gamma_X;

    X_L1 = sparse(X_opt.regul_tr1*L);
    X_L2 = sparse(X_opt.regul_tr2*(eye(size(G_gf,1),size(G_gf,1)) - Normal_M_modified(G_gf)));
    
    % for Y
    Y_opt = m_copt;
    Y_opt.gamma = zeta;
    Y_opt.regul_tr1 = beta_Y;
    Y_opt.regul_tr2 = gamma_Y;
    
    Y_L1 = sparse(Y_opt.regul_tr1*L);
    Y_L2 = sparse(Y_opt.regul_tr2*(eye(size(G_ph,1),size(G_ph,1)) - Normal_M_modified(G_ph)));
       
    first_obj_val_X = -1;
    old_obj_val_X = -1;
    first_obj_val_Y = -1;
    old_obj_val_Y = -1;
    should_continue_X = true;
    should_continue_Y = true;
    
    for m_nt = 1:m_nMaxturns,
        
        %- Update X:
        if should_continue_X
            [X_sol,m_vobjs] = myfunc_lbfgsb_tlDLP(X', X0', Z0, Y', X_L1, X_L2, X_opt);
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
            [Y_sol, m_vobjs] = myfunc_lbfgsb_tlDLP(Y', Y0', Z0', X', Y_L1, Y_L2, Y_opt);
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
    m_cResult.tlDLP_Xhat = X;
    m_cResult.tlDLP_Yhat = Y;
                
    %- Evaluation for Multi-Task GL
    m_vNewIDXR = m_vidxR(mv_TstIDX);
    m_vNewIDXC = m_vIdxC(mv_TstIDX);
    
    mv_uniqIRs = unique(m_vNewIDXR);
    
    m_vTotRanks = [];
    for mn_i = 1:length(mv_uniqIRs),
        mn_pos = mv_uniqIRs(mn_i);
            
        if sum(G_ph(:,mn_pos)) ~= 1 % only leaf
            continue;
        end
        
        m_vTrueIDX = find(Y0_TST(mn_pos,:));
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
    
    m_cResult.tlDLP_RankAvg = mean(m_vTotRanks);
    m_cResult.tlDLP_Eval = m_vTotRanks;

end

