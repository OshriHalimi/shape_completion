function [] = visualize_conncomp(M)

% M = test_mesh('car_175k.off');  
% M = test_mesh('voyager_7kDsb.ply'); 
[N,I,cnts] = conn_comp(M.A);
clrs = shuffle(hsv(N)); 
vclrs = zeros(M.Nv,3);
for i=1:N
    vclrs(I==i,:) = repmat(clrs(i,:),cnts(i),1); 
end
M.plt.title = sprintf('#Connected Component = %d',N); 
M.ezvisualize(vclrs); 
end
