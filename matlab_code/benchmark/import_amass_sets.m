function [gt_set,res_set] = import_amass_sets(target_exp)

tgt_res_dir = fullfile(up_script_dir(1),'data',target_exp); 
if ~exist(tgt_res_dir, 'dir')
    assert(false,sprintf("Warning - Could not find experiment repository at %s",tgt_res_dir))
end

res_fps = findfiles(tgt_res_dir,{'*.mat'});
N_res = numel(res_fps);
gt_set = res_fps
% progressbar;
% for i=1:N_mfp
%     ds.ms{i} = read_mesh(mfp{i}); progressbar(i/N_mfp);
% end



end

        function [ds]= MeshDataset(name,force)
            ds.name = lower(name);
            ds.path=name2path(ds.name);
            ds.cache_path = fullfile(ds.path,sprintf('%s_ds.mat',ds.name));
            
            if ~(exist('force','var') && force) && isfile(ds.cache_path)
                load(ds.cache_path,'ds');
            else % Load ALL meshes found (TODO - sparse it)
                
                mfp = mfp(~ismember(mfp,ds.cache_path)); % Remove cache file
                N_mfp = numel(mfp); ds.ms = cell(N_mfp,1);  progressbar;
                for i=1:N_mfp
                    ds.ms{i} = read_mesh(mfp{i}); progressbar(i/N_mfp);
                end
                ds.N = numel(ds.ms);
                ds = parse_classes(ds,mfp);
                ds = ds.validate();
                ds.cache();
            end
            fprintf('Successfully loaded dataset %s with %d meshes and %d classes\n',upper(name),ds.N,ds.Nc);
        end