function [f,f_ds] = get_representative_tri(c)
M = read_mesh(fullfile(c.path.collat_dir,'face_ref.OFF')); 
f = M.f;
M = read_mesh(fullfile(c.path.collat_dir,'downsampled_tri.off')); 
f_ds = M.f;
end