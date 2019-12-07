function [res,gt,part,tp,mask] = load_path_tup(c,tup_record)

% File path creation
res_fp =  fullfile(c.path.curr_exp_dir,tup_record{1});
part_fp = fullfile(c.path.projection_dir,tup_record{2});
gt_fp = fullfile(c.path.full_shape_dir,tup_record{3});
temp_fp = fullfile(c.path.full_shape_dir,tup_record{4});

% Load the GT, Template and Mask
switch c.curr_tgt_ds
    case 'faust'
        gt = load(gt_fp); tp = load(temp_fp); part = load(part_fp);
        tp = double(tp.full_shape(:,1:3)); gt = double(gt.full_shape(:,1:3));
        mask = part.part_mask;
        tp = Mesh(tp,c.f,'Template');
        gt = Mesh(gt,c.f,'Ground Truth');
    case 'amass'
        gt = read_mesh(gt_fp,'Ground Truth');
        tp = read_mesh(temp_fp,'Template');
        tgt_unzip_dir = fullfile(c.path.tmp_dir,get_tid()); 
        if ~isfolder(tgt_unzip_dir)
            mkdir(tgt_unzip_dir);
        end
        unzip(part_fp,tgt_unzip_dir);
        mask = double(readNPY(fullfile(tgt_unzip_dir,'mask.npy'))+1);
end

% Compute Part from Mask
dead_verts = 1:gt.Nv; dead_verts(mask) = [];
part = remove_vertices(gt,dead_verts);
part.name = 'Partial Shape'; part.plt.title = 'Partial Shape';

% Load the Result
switch c.curr_src_ds
    case '3d-coded'
        res = read_mesh(res_fp,'Partial');
        res.v = res.v * nthroot(0.0638/res.volume(),3);
    case {'farm','3d-epn'}
        res = read_mesh(res_fp,'Partial');
    case {'litany'}
        res = load(res_fp);
        res = Mesh(squeeze(res.output_vertices),c.f_ds,'Result'); 
    otherwise
        res = load(res_fp);
        res = double(squeeze(res.pointsReconstructed(:,1:3,:)).');
        res = Mesh(res,c.f,'Result');
end
end

%     Nsegs = conn_comp(partM.A);
%     if Nsegs > 1
%         n_disconnected = n_disconnected +1;
%         fprintf('%d : Num disconnected comp %d\n',n_disconnected,Nsegs);
%         visualize_conncomp(partM)
%     end
