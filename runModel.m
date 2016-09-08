function runModel(optionsEvaluationOnt, optionsModel, ...
    params, optionsGO, output_filepath, fold_start, fold_end, train_model, ...
    go_filepath, hpo_filepath, ppi_filepath, cv_index_filepath, tdlp_Y0, max_depth_children)

    if strcmp(optionsGO, 'MF')
        mstr_dataset = 'molecular_function';
    else
        mstr_dataset = 'biological_process';
    end

    % load ontologies
    tmp1 = load(go_filepath);
    tmp2 = load(hpo_filepath);

    % the gene list in tmp1
    m_cgeneListG1 = tmp1.gene_table;

    % the gene list in tmp2
    m_cgeneListG2 = cell(length(tmp2.gene_table),1);
    for m_ni = 1:length(tmp2.gene_table),
        m_cgeneListG2{m_ni} = char(tmp2.gene_table{m_ni});
    end

    % genes in the intersection
    m_vINDEX = sum(tmp2.profiles,2) > 0;
    m_cgeneListG2_sub = m_cgeneListG2(m_vINDEX);
    tmp2_profiles_sub = tmp2.profiles(m_vINDEX,:)';
    
    [m_cGeneList, m_vAidx_gf, ~] = ...
            intersect(upper(m_cgeneListG1), upper(m_cgeneListG2_sub)); %
    if strcmp(optionsGO, 'MF')
        Xtrue = tmp1.profiles(m_vAidx_gf,:)';
    end

    Ytrue = zeros(size(tmp2.profiles,2), length(m_cGeneList));
    [~, m_vAidx_ph, m_vBidx_ph] = ...
            intersect(upper(m_cGeneList), upper(m_cgeneListG2_sub)); %
    Ytrue(:,m_vAidx_ph) = tmp2_profiles_sub(:,m_vBidx_ph);
        
    % m: the number of the genes
    m = length(m_cGeneList);
    m_vGeneIDXE = 1:m;

    % S: normalized PPI network generated from inverse graphallshortestpaths
    dist = load(ppi_filepath);

    dist = dist.dist;
    dist = sparse(dist);
    S = Normal_M_modified(dist);

    % L: graph Laplacian
    L = eye(m,m) - S;

    % gene function 
    G_gf = tmp1.groups > 0;
    r_gf = 1./sum(G_gf,2)';
    d_gf = tmp1.depth;

    % phenotype
    G_ph = tmp2.groups > 0;
    r_ph = 1./sum(G_ph,2)';
    d_ph = tmp2.depth;

    mc_CVset = load(cv_index_filepath);
    mc_CVset = mc_CVset.mc_CVindex;

    fprintf('Data set: %s\n', mstr_dataset);
    fprintf('Checking: #genes = %d \n', m)

    %- setting
    m_copt.maxIts = 1;
    m_copt.maxTotalIts = 200;

    m_copt.m = 5;
    m_copt.factor =  1e7;
    m_copt.pgtol = 1e-5;

    m_copt.verbose = 2;

    m_copt.mu = 0.1;

    VERBOSE = 2; % 0: silent, 1: outer iterations, 2: + inner interations 

    m_nMaxturns = 20;

    m_nNumTrials = 10;

    m_cResult = cell(m_nNumTrials,1);

    for m_ntrial = fold_start:fold_end
        fprintf('%2dth fold \n', m_ntrial);    

        %- initial solutions 
        X0 = Xtrue;
        Y0 = Ytrue;    

        mv_TrnIDX = mc_CVset{m_ntrial}.trnIDX;
        mv_TstIDX = mc_CVset{m_ntrial}.tstIDX;
        
        mv_CVTrnIDX = mc_CVset{m_ntrial}.CVtrnIDX;
        mv_CVTstIDX = mc_CVset{m_ntrial}.CVtstIDX;
        
        if train_model
            mv_CVTrnIDX = union(mv_CVTrnIDX,mv_CVTstIDX);
            mv_TrnIDX = union(mv_TrnIDX,mv_TstIDX);
        end
        
        % Y0_CV
        if strcmp(optionsEvaluationOnt, 'HPO')
            [m_vidxR, m_vIdxC, m_vValT] = find(Ytrue);
            
            m_vNewVals = m_vValT(mv_CVTrnIDX);
            m_vNewIDXR = m_vidxR(mv_CVTrnIDX);
            m_vNewIDXC = m_vIdxC(mv_CVTrnIDX);
            Y0_CV = sparse(m_vNewIDXR,m_vNewIDXC,m_vNewVals,size(G_ph,1),m);
            Y0_CV = removeChildrenAssoc(G_ph, Y0_CV, d_ph, max_depth_children);

            m_vNewVals = m_vValT(mv_TrnIDX);
            m_vNewIDXR = m_vidxR(mv_TrnIDX);
            m_vNewIDXC = m_vIdxC(mv_TrnIDX);
            Y0_TST = sparse(m_vNewIDXR,m_vNewIDXC,m_vNewVals,size(G_ph,1),m);
            Y0_TST = removeChildrenAssoc(G_ph, Y0_TST, d_ph, max_depth_children);
        else
            [m_vidxR, m_vIdxC, m_vValT] = find(Xtrue);
            
            m_vNewVals = m_vValT(mv_CVTrnIDX);
            m_vNewIDXR = m_vidxR(mv_CVTrnIDX);
            m_vNewIDXC = m_vIdxC(mv_CVTrnIDX);
            X0_CV = sparse(m_vNewIDXR,m_vNewIDXC,m_vNewVals,size(G_gf,1),m);
            X0_CV = removeChildrenAssoc(G_gf, X0_CV, d_gf, max_depth_children);

            m_vNewVals = m_vValT(mv_TrnIDX);
            m_vNewIDXR = m_vidxR(mv_TrnIDX);
            m_vNewIDXC = m_vIdxC(mv_TrnIDX);
            X0_TST = sparse(m_vNewIDXR,m_vNewIDXC,m_vNewVals,size(G_gf,1),m);
            X0_TST = removeChildrenAssoc(G_gf, X0_TST, d_gf, max_depth_children);
        end

        m_cResult{m_ntrial}.datastr = mstr_dataset;
        m_cResult{m_ntrial}.trnIDX = mv_TrnIDX;    
        m_cResult{m_ntrial}.tstIDX = mv_TstIDX;
        m_cResult{m_ntrial}.CVtrnIDX = mv_CVTrnIDX;
        m_cResult{m_ntrial}.CVtstIDX = mv_CVTstIDX;

        % DLP 
        if strcmp(optionsModel, 'DLP')
            %------------------------- DLP ---------------------------------%

            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_DLP = DLP_CV(Y0_CV, params, ...
                    m_vidxR, m_vIdxC, length(m_vGeneIDXE), ...
                    mv_CVTstIDX, G_ph, m_copt, L);
            else
                m_cResult_DLP = DLP_CV(X0_CV, params, ...
                    m_vidxR, m_vIdxC, length(m_vGeneIDXE), ...
                    mv_CVTstIDX, G_gf, m_copt, L);
            end
            
            m_cResult{m_ntrial}.DLP_CVresult = m_cResult_DLP.DLP_CVresult;
            m_cResult{m_ntrial}.DLP_BParams = m_cResult_DLP.DLP_BParams;
            m_cResult{m_ntrial}.DLP_TotRanks = m_cResult_DLP.DLP_TotRanks;

            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_DLP = DLP(m_cResult_DLP.DLP_BParams, Y0_TST, ...
                    m_vidxR, m_vIdxC, ...
                    length(m_vGeneIDXE), mv_TstIDX, ...
                    G_ph, m_copt, L);
            else
                m_cResult_DLP = DLP(m_cResult_DLP.DLP_BParams, X0_TST, ...
                    m_vidxR, m_vIdxC, ...
                    length(m_vGeneIDXE), mv_TstIDX, ...
                    G_gf, m_copt, L);
            end

            m_cResult{m_ntrial}.DLP_Yhat = m_cResult_DLP.DLP_Yhat;
            m_cResult{m_ntrial}.DLP_RankAvg = m_cResult_DLP.DLP_RankAvg;
            m_cResult{m_ntrial}.DLP_Eval = m_cResult_DLP.DLP_Eval;
            m_vTotRanks = m_cResult{m_ntrial}.DLP_Eval;

            fprintf('[%d fold:: DLP] Avg rank = %3.3f \n', m_ntrial, mean(m_vTotRanks));    
            fprintf('Total number of associations examined: %d \n', length(m_vTotRanks));
            fprintf('Number of associations ranked     r  <=   5: %3d \n', sum(m_vTotRanks<=5));
            fprintf('Number of associations ranked 5  < r <=  10: %3d \n', sum(m_vTotRanks> 5 & m_vTotRanks<=10));
            fprintf('Number of associations ranked 10 < r <=  50: %3d \n', sum(m_vTotRanks>10 & m_vTotRanks<=50));
            fprintf('Number of associations ranked 50 < r <= 100: %3d \n', sum(m_vTotRanks>50 & m_vTotRanks<=100));    
        end
        
        % OGL 
        if strcmp(optionsModel, 'OGL')
            %------------------------- OGL ---------------------------------%

            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_OGL = OGL_CV(Y0_CV, params, ...
                    m_vidxR, m_vIdxC, length(m_vGeneIDXE), ...
                    mv_CVTstIDX, G_ph, r_ph, m_copt, L);
            else
                m_cResult_OGL = OGL_CV(X0_CV, params, ...
                    m_vidxR, m_vIdxC, length(m_vGeneIDXE), ...
                    mv_CVTstIDX, G_gf, r_gf, m_copt, L);
            end
            
            m_cResult{m_ntrial}.OGL_CVresult = m_cResult_OGL.OGL_CVresult;
            m_cResult{m_ntrial}.OGL_BParams = m_cResult_OGL.OGL_BParams;
            m_cResult{m_ntrial}.OGL_TotRanks = m_cResult_OGL.OGL_TotRanks;

            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_OGL = OGL(m_cResult_OGL.OGL_BParams, Y0_TST, ...
                    m_vidxR, m_vIdxC, ...
                    length(m_vGeneIDXE), mv_TstIDX, ...
                    G_ph, r_ph, m_copt, L);
            else
                m_cResult_OGL = OGL(m_cResult_OGL.OGL_BParams, X0_TST, ...
                    m_vidxR, m_vIdxC, ...
                    length(m_vGeneIDXE), mv_TstIDX, ...
                    G_gf, r_gf, m_copt, L);
            end

            m_cResult{m_ntrial}.OGL_Yhat = m_cResult_OGL.OGL_Yhat;
            m_cResult{m_ntrial}.OGL_RankAvg = m_cResult_OGL.OGL_RankAvg;
            m_cResult{m_ntrial}.OGL_Eval = m_cResult_OGL.OGL_Eval;
            m_vTotRanks = m_cResult{m_ntrial}.OGL_Eval;

            fprintf('[%d fold:: OGL] Avg rank = %3.3f \n', m_ntrial, mean(m_vTotRanks));    
            fprintf('Total number of associations examined: %d \n', length(m_vTotRanks));
            fprintf('Number of associations ranked     r  <=   5: %3d \n', sum(m_vTotRanks<=5));
            fprintf('Number of associations ranked 5  < r <=  10: %3d \n', sum(m_vTotRanks> 5 & m_vTotRanks<=10));
            fprintf('Number of associations ranked 10 < r <=  50: %3d \n', sum(m_vTotRanks>10 & m_vTotRanks<=50));
            fprintf('Number of associations ranked 50 < r <= 100: %3d \n', sum(m_vTotRanks>50 & m_vTotRanks<=100));    
        end

        % tlDLP 
        if strcmp(optionsModel, 'tlDLP')
            
            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_tlDLP = tlDLP_CV(X0, Y0_CV, params, m, S, m_copt, ...
                    L, m_nMaxturns, G_gf, G_ph, ...
                    m_vidxR, m_vIdxC, mv_CVTstIDX, m_vGeneIDXE, tdlp_Y0);
            else
                m_cResult_tlDLP = tlDLP_CV(Y0, X0_CV, params, m, S, m_copt, ...
                    L, m_nMaxturns, G_ph, G_gf, ...
                    m_vidxR, m_vIdxC, mv_CVTstIDX, m_vGeneIDXE, tdlp_Y0);
            end
            
            m_cResult{m_ntrial}.tlDLP_CVresult = m_cResult_tlDLP.tlDLP_CVresult;
            m_cResult{m_ntrial}.tlDLP_BParams = m_cResult_tlDLP.tlDLP_BParams;
            m_cResult{m_ntrial}.tlDLP_TotRanks = m_cResult_tlDLP.tlDLP_TotRanks;

            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_tlDLP = tlDLP(X0, m_cResult_tlDLP.tlDLP_BParams, m, S, ...
                    m_copt, L, m_nMaxturns, G_gf, G_ph, ...
                    m_vValT, m_vidxR, m_vIdxC, mv_TstIDX, ...
                    m_vGeneIDXE, mv_TrnIDX, tdlp_Y0);
            else
                m_cResult_tlDLP = tlDLP(Y0, m_cResult_tlDLP.tlDLP_BParams, m, S, ...
                    m_copt, L, m_nMaxturns, G_ph, G_gf, ...
                    m_vValT, m_vidxR, m_vIdxC, mv_TstIDX, ...
                    m_vGeneIDXE, mv_TrnIDX, tdlp_Y0);
            end

            m_cResult{m_ntrial}.tlDLP_Xhat =  m_cResult_tlDLP.tlDLP_Xhat;
            m_cResult{m_ntrial}.tlDLP_Yhat =  m_cResult_tlDLP.tlDLP_Yhat;
            m_cResult{m_ntrial}.tlDLP_RankAvg = m_cResult_tlDLP.tlDLP_RankAvg;
            m_cResult{m_ntrial}.tlDLP_Eval = m_cResult_tlDLP.tlDLP_Eval;
            m_vTotRanks = m_cResult{m_ntrial}.tlDLP_Eval;

            fprintf('[%d fold:: tlDLP] Avg rank = %3.3f \n', m_ntrial, mean(m_vTotRanks));
            fprintf('Total number of associations examined: %d \n', length(m_vTotRanks));
            fprintf('Number of associations ranked      r <=   5: %3d \n', sum(m_vTotRanks<=5));
            fprintf('Number of associations ranked 5  < r <=  10: %3d \n', sum(m_vTotRanks> 5 & m_vTotRanks<=10));
            fprintf('Number of associations ranked 10 < r <=  50: %3d \n', sum(m_vTotRanks>10 & m_vTotRanks<=50));
            fprintf('Number of associations ranked 50 < r <= 100: %3d \n', sum(m_vTotRanks>50 & m_vTotRanks<=100));
        end

        % LP
        if strcmp(optionsModel, 'LP')
            %------------------- Base line method ---------------------%
            %------------------- Label propagation --------------------%
            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_LP = LP(params, Y0_CV, Y0_TST, ...
                    m_vidxR, m_vIdxC, length(m_vGeneIDXE), mv_CVTstIDX, mv_TstIDX, ...
                    m, S);
            else
                m_cResult_LP = LP(params, X0_CV, X0_TST, ...
                    m_vidxR, m_vIdxC, length(m_vGeneIDXE), mv_CVTstIDX, mv_TstIDX, ...
                    m, S);
            end

            m_cResult{m_ntrial}.LP_CVresult = m_cResult_LP.LP_CVresult;
            m_cResult{m_ntrial}.LP_BParams = m_cResult_LP.LP_BParams;
            m_cResult{m_ntrial}.LP_Yhat =  m_cResult_LP.LP_Yhat;
            m_cResult{m_ntrial}.LP_RankAvg = m_cResult_LP.LP_RankAvg;
            m_cResult{m_ntrial}.LP_Eval = m_cResult_LP.LP_Eval;
            m_vTotRanks = m_cResult{m_ntrial}.LP_Eval;

            fprintf('[%d fold:: LP] Avg rank = %3.3f \n', m_ntrial, mean(m_vTotRanks));    
            fprintf('Total number of associations examined: %d \n', length(m_vTotRanks));
            fprintf('Number of associations ranked      r <=   5: %3d \n', sum(m_vTotRanks<=5));
            fprintf('Number of associations ranked 5  < r <=  10: %3d \n', sum(m_vTotRanks> 5 & m_vTotRanks<=10));
            fprintf('Number of associations ranked 10 < r <=  50: %3d \n', sum(m_vTotRanks>10 & m_vTotRanks<=50));
            fprintf('Number of associations ranked 50 < r <= 100: %3d \n', sum(m_vTotRanks>50 & m_vTotRanks<=100));    
        end

        % BiRW
        if strcmp(optionsModel, 'BiRW')
            %------------------- Base line method ---------------------%
            %------------------- BiRW -------------------------------%
            if strcmp(optionsEvaluationOnt, 'HPO')
                m_cResult_BiRW = BiRW(params, Y0_CV, Y0_TST, ...
                Normal_M_modified(G_ph), m_vidxR, m_vIdxC, ...
                length(m_vGeneIDXE), mv_TstIDX, S);
            else
                m_cResult_BiRW = BiRW(params, X0_CV, X0_TST, ...
                Normal_M_modified(G_gf), m_vidxR, m_vIdxC, ...
                length(m_vGeneIDXE), mv_TstIDX, S);
            end

            m_cResult{m_ntrial}.BiRW_CVresult = m_cResult_BiRW.BiRW_CVresult;
            m_cResult{m_ntrial}.BiRW_BParams = m_cResult_BiRW.BiRW_BParams;
            m_cResult{m_ntrial}.BiRW_Yhat =  m_cResult_BiRW.BiRW_Yhat;
            m_cResult{m_ntrial}.BiRW_RankAvg = m_cResult_BiRW.BiRW_RankAvg;
            m_cResult{m_ntrial}.BiRW_Eval = m_cResult_BiRW.BiRW_Eval;
            m_vTotRanks = m_cResult{m_ntrial}.BiRW_Eval;

            fprintf('[%d fold:: BiRW] Avg rank = %3.3f \n', m_ntrial, mean(m_vTotRanks));    
            fprintf('Total number of associations examined: %d \n', length(m_vTotRanks));
            fprintf('Number of associations ranked      r <=   5: %3d \n', sum(m_vTotRanks<=5));
            fprintf('Number of associations ranked 5  < r <=  10: %3d \n', sum(m_vTotRanks> 5 & m_vTotRanks<=10));
            fprintf('Number of associations ranked 10 < r <=  50: %3d \n', sum(m_vTotRanks>10 & m_vTotRanks<=50));
            fprintf('Number of associations ranked 50 < r <= 100: %3d \n', sum(m_vTotRanks>50 & m_vTotRanks<=100));    
        end

        %--- save results
        [pathstr,name,ext] = fileparts(output_filepath);
        save(strcat(pathstr, '/', name, '_fold', num2str(m_ntrial), ext), 'm_cResult');
    end
end

