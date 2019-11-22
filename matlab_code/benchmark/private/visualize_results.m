function visualize_results(c,stats,paths) % TODO: Handle more than Faust

if c.n_renders <= 0; return; end

if c.export_render_to_ply || c.write_gif
    c.path.exp_res_dir = fullfile(c.path.exps_dir,sprintf('%s_Visualizations',c.curr_exp));
    if ~isfolder(c.path.exp_res_dir)
        mkdir(c.path.exp_res_dir)
        fprintf('Created visualizations directiory %s\n',c.path.exp_res_dir);
    end
end

if c.write_gif
    v = VideoWriter(fullfile( c.path.exp_res_dir,sprintf('%s_GIF',c.exp_name)));
    v.FrameRate = c.frame_rate; v.Quality = 100;
    open(v); fullfig;
end

N = min(size(paths,1),c.n_renders);

if c.cherry_pick_mode
    living = false(N,1);
    cherries = 0; 
end


for i=1:N
    
    if ~c.write_gif
        fullfig; % Generate new figure every time
    else
        clf; % Clear the single figure
    end
    
    [resM,gtM,partM,tpM] = load_path_tup(c,paths(i,:));
    visualize_meshes(c,gtM,tpM,partM,resM,paths{i,1});
    
    if c.cherry_pick_mode
        choice = questdlg(sprintf('%d/%d: Is this a good result?',i,N),'Cherry Picker','Yes','No', 'Yes');
        if strcmp(choice,'Yes')
            living(i) = true;
            if c.export_render_to_ply
                cherries = cherries +1;
                resM.export_as(fullfile(c.path.exp_res_dir,sprintf('res_%d.ply',cherries)));
                gtM.export_as(fullfile(c.path.exp_res_dir,sprintf('gt_%d.ply',cherries)));
                partM.export_as(fullfile(c.path.exp_res_dir,sprintf('part_%d.ply',cherries)));
                tpM.export_as(fullfile(c.path.exp_res_dir,sprintf('tp_%d.ply',cherries)));
            end
        end
    end
    
    if c.write_gif
        writeVideo(v,getframe(gcf));
    end
end

if c.write_gif; close(v); end

if c.cherry_pick_mode
    
    len = size(paths,1);
    paths = paths(living,:);
    if isempty(paths)
        warning('Found no cherries to import')
        return;
    end
    stats = trunc_struct(stats,len,living);
    fprintf('Exporting %d cherries\n',nnz(living));
    export_subset(c,paths,stats);
end
end


function visualize_meshes(c,gtM,tpM,partM,resM,res_name)
% suptitle(uplw([c.exp_name,' ',res_name])); axis('off');
% oplt.new_fig=0; oplt.disp_ang = [0,90]; oplt.limits = 0;
% subplot_tight(1,6,1);
% resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
% subplot_tight(1,6,2);
% gtM.visualize_vertices(uniclr('teal',gtM.Nv),oplt);
% subplot_tight(1,6,3);
% resM.visualize_vertices(uniclr('teal',resM.Nv),oplt);
% partM.visualize_vertices(uniclr('r',partM.Nv),oplt);
% subplot_tight(1,6,4);
% partM.visualize_vertices(uniclr('teal',partM.Nv),oplt);
% subplot_tight(1,6,5);
% tpM.visualize_vertices(uniclr('teal',tpM.Nv),oplt);
% subplot_tight(1,6,6);
% resM.ezvisualize([],oplt);

% oplt.disp_func = @visualize_vertices;
% mesh_plot_pair(resM,gtM,uniclr('teal',resM.Nv),uniclr('teal',resM.Nv),oplt);
% oplt.disp_func = @ezvisualize;
% mesh_plot_pair(resM,gtM,uniclr('w',resM.Nv),uniclr('w',resM.Nv),oplt);
% mesh_plot_pair(resM,tpM,uniclr('w',resM.Nv),uniclr('w',resM.Nv),oplt);

suptitle(uplw([c.curr_exp,' ',res_name])); axis('off');
oplt.new_fig=0; oplt.disp_ang = [0,90]; oplt.limits = 0;
subplot_tight(1,4,1);
resM.ezvisualize([],oplt);
subplot_tight(1,4,2);
gtM.ezvisualize([],oplt);
subplot_tight(1,4,3);
partM.ezvisualize([],oplt);
subplot_tight(1,4,4);
tpM.ezvisualize([],oplt);
end