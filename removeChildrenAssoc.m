function Y0 = removeChildrenAssoc(G_ph, Y0, d_ph, max_depth)
    for node_idx = 1:size(G_ph,1)
        if d_ph(node_idx) > max_depth
            zero_on_node = find(Y0(node_idx,:)==0);
            children_nodes = getChildrenNodes(G_ph,node_idx);

            % apply zeros to children
            Y0(children_nodes,zero_on_node) = 0;
        end
    end
end

