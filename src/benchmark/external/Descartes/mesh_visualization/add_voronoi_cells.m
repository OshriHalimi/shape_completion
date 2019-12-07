function [] = add_voronoi_cells(M,vi,ve)
add_feature_pts(M,M.v(vi,:),0.03,'w');
hold on;
for i=1:size(ve,2)
    plot3( [ve(1,i) ve(4,i)], [ve(2,i) ve(5,i)], [ve(3,i) ve(6,i)], 'w','LineWidth',2);
end
hold off;
end

