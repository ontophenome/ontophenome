function m_vTotRanks = BiRW_Eval(Y0, Y, phen_idxs, genes_idxs, n_genes, tst_idx, G_ph)

    m_vNewIDXR = phen_idxs(tst_idx);
    m_vNewIDXC = genes_idxs(tst_idx);

    mv_uniqIRs = unique(m_vNewIDXR);   

    m_vTotRanks = [];
    for mn_i = 1:length(mv_uniqIRs),
        mn_pos = mv_uniqIRs(mn_i);

        if sum(G_ph(:,mn_pos)) ~= 1 % only leaf
            continue;
        end
        
        m_vTrueIDX = find(Y0(mn_pos,:));
        m_vCandIDX = m_vNewIDXC(m_vNewIDXR==mn_pos);  

        m_vCurGeneIDXE = 1:n_genes;
        m_vCurGeneIDXE([m_vTrueIDX, m_vCandIDX']) = [];

        m_vRank = zeros(length(m_vCandIDX), 1);
        for mn_j = 1:length(m_vCandIDX),
            m_vEstVals = full(Y(mn_pos, [m_vCandIDX(mn_j), m_vCurGeneIDXE]));      
            [~, mv_idxs] = sort(m_vEstVals, 2, 'descend');

            m_vRank(mn_j) = find(mv_idxs == 1) + sum(m_vEstVals(1)==m_vEstVals) - 1;
        end

        m_vTotRanks = [m_vTotRanks; m_vRank]; %#ok<AGROW>           
    end

end
