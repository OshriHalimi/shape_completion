function [edge_plots] = add_geodesic_paths(M,paths,clrs)

if ~exist('clrs','var'); clrs = 'c'; end
if ~iscell(paths)
    paths = {paths};
end

edge_plots = zeros(1,numel(paths)); %For legend 
hold on;
for i=1:numel(paths)
    P = paths{i};
    if iscell(clrs)
        clr = clrs{i}; 
    else
        clr = clrs; 
    end
    if size(P,1)==1 % Vertices: 
        add_feature_pts(M,[M.v(P(1),:);M.v(P(end),:)]);
        E =[P(1:end-1);P(2:end)].';
        AP = add_edge_visualization(M,E,0,clr); 
        edge_plots(i) = AP(1); 
    else % Geodesic
        add_feature_pts(M,[P(:,1),P(:,end)].');
        hold on; 
        AP = plot3(P(1,:), P(2,:),P(3,:), clr,'LineWidth',2);
        hold off; 
        edge_plots(i) = AP(1); 
    end
end
hold off; 

