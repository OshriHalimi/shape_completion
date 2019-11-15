
function visualize_meshes(c,gtM,tpM,partM,resM,res_name)
suptitle(uplw([c.exp_name,' ',res_name])); axis('off');
oplt.new_fig=0; oplt.disp_ang = [0,90]; oplt.limits = 0;
subplot_tight(1,6,1);
resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
subplot_tight(1,6,2);
gtM.visualize_vertices(uniclr('teal',gtM.Nv),oplt);
subplot_tight(1,6,3);
resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
subplot_tight(1,6,4);
partM.visualize_vertices(uniclr('teal',partM.Nv),oplt);
subplot_tight(1,6,5);
tpM.visualize_vertices(uniclr('teal',tpM.Nv),oplt);
subplot_tight(1,6,6);
resM.ezvisualize([],oplt);

% oplt.disp_func = @visualize_vertices;
% mesh_plot_pair(resM,gtM,uniclr('teal',resM.Nv),uniclr('teal',resM.Nv),oplt);
% oplt.disp_func = @ezvisualize;
% mesh_plot_pair(resM,gtM,uniclr('w',resM.Nv),uniclr('w',resM.Nv),oplt);
% mesh_plot_pair(resM,tpM,uniclr('w',resM.Nv),uniclr('w',resM.Nv),oplt);
end