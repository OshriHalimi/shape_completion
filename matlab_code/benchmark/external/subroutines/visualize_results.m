function visualize_results(c,paths) % TODO: Handle more than Faust

if c.export_render_to_ply || c.write_gif
    c.exp_res_dir = fullfile(c.exp_dir,sprintf('%s_Visualizations',c.exp_name));
    if ~isfolder(c.exp_res_dir)
        mkdir(c.exp_res_dir)
        fprintf('Created visualizations directiory %s\n',c.exp_res_dir);
    end
end

if c.write_gif
    v = VideoWriter(fullfile( c.exp_res_dir,sprintf('%s_GIF',c.exp_name)));
    v.FrameRate = c.frame_rate; v.Quality = 100;
    open(v); fullfig;
end

for i=1:min(size(paths,1),c.n_renders)
    
    [resM,gtM,partM,tpM] = load_path_tup(c,paths(i,:));
    if ~c.write_gif
        fullfig; % Generate new figure every time
    else
        clf; % Clear the single figure
    end
    visualize_meshes(c,gtM,tpM,partM,resM,paths{i,1});
    
    if c.export_render_to_ply
        resM.export_as(fullfile(c.exp_res_dir,sprintf('res_%d.ply',i)));
        gtM.export_as(fullfile(c.exp_res_dir,sprintf('gt_%d.ply',i)));
        partM.export_as(fullfile(c.exp_res_dir,sprintf('part_%d.ply',i)));
        tpM.export_as(fullfile(c.exp_res_dir,sprintf('tp_%d.ply',i)));
    end
    if c.write_gif
        writeVideo(v,getframe(gcf));
    end
end

if c.write_gif; close(v); end

end