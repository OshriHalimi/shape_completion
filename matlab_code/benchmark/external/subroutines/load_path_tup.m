function [res,gt,part,tp,mask] = load_path_tup(c,tup_record)

% File path creation
res_fp =  fullfile(c.curr_exp_dir,tup_record{1});
part_fp = fullfile(c.projection_dir,tup_record{2});
gt_fp = fullfile(c.full_shape_dir,tup_record{3});
temp_fp = fullfile(c.full_shape_dir,tup_record{4});
% parttrivfp = fullfile(c.triv_dir,tup_record{2});

% Loading
switch c.ds_name
    case 'faust'
        gt = load(gt_fp); tp = load(temp_fp); part = load(part_fp);
        tp = double(tp.full_shape(:,1:3)); gt = double(gt.full_shape(:,1:3));
        mask = part.part_mask;
        tp = Mesh(tp,c.f,'Template');
        gt = Mesh(gt,c.f,'Ground Truth');
    case 'amass'
        gt = read_mesh(gt_fp,'Ground Truth');
        tp = read_mesh(temp_fp,'Template');
        if isfile('mask.npy') % Just to make sure
            delete('mask.npy'); 
        end
        unzip(part_fp); % Warning: Dangerous to Parfor this 
        mask = readNPY('mask.npy')+1;
end
dead_verts = 1:gt.Nv;
dead_verts(mask) = [];
part = remove_vertices(gt,dead_verts);
part.name = 'Partial Shape';
part.plt.title = 'Partial Shape';
res = load(res_fp);
res = double(squeeze(res.pointsReconstructed(:,1:3,:)).');
res = Mesh(res,c.f,'Result');

%     Nsegs = conn_comp(partM.A);
%     if Nsegs > 1
%         n_disconnected = n_disconnected +1;
%         fprintf('%d : Num disconnected comp %d\n',n_disconnected,Nsegs);
%         visualize_conncomp(partM)
%     end

end