function add_segmentation_edges(M,seg,o)
for i=1:numel(unique(seg))
    if numel(seg) == M.Nv
        % Transform vertices -> faces
        vi = find(seg==i);
        fi = M.neighbors(vi,'vf');
        % Now we need to remove all faces that are not of the current color
    else
        fi = find(seg==i);
    end
    be = M.boundary(fi);
    if iscell(o.edge_clr)
        add_edge_visualization(M,be,0,uniclr(o.edge_clr{i}));
    else
        add_edge_visualization(M,be,0,o.edge_clr);
    end
end
end