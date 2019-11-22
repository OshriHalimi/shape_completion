function export_subset(c,paths,stats)

N = size(paths,1); 

% Set EXP 
dset.name = c.curr_exp;
dset.stats = stats; 
dset.meshes_are_ordered_by = c.sort_meth;

dset.fvs = cell(N+1,5);
dset.fvs{1,1} = 'Result';
dset.fvs{1,2} = 'Ground Truth';
dset.fvs{1,3} = 'Part';
dset.fvs{1,4} = 'Template';
dset.fvs{1,5} = 'Mask';

for i=1:N
    [resM,gtM,partM,tempM,mask] = load_path_tup(c,paths(i,:));
    resM.v = compute_icp(resM.v,gtM.v,true);
    partM.v = compute_icp(partM.v,resM.v,true);
    
    dset.fvs{i+1,1} = resM.fv_struct();
    dset.fvs{i+1,2} = gtM.fv_struct();
    dset.fvs{i+1,3} = partM.fv_struct();
    dset.fvs{i+1,4} = tempM.fv_struct();
    dset.fvs{i+1,5} = mask;
    
end

dset_fp = fullfile(up_script_dir(0),sprintf('%s_%d_dset.mat',c.curr_exp,N)); 
save(dset_fp,'dset','-v7.3');
fprintf('Exported subset to %s\n',dset_fp); 
end