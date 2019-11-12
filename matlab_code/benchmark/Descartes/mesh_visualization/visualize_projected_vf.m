function visualize_projected_vf(M,vf,normalize,plt_override)

if size(vf,2) == 1
    vf = reshape(vf,M.Nf,3);
end

if ~exist('normalize','var') ; normalize = 1; end
if ~exist('plt_override','var'); plt_override = struct(); end

[vfp,f] = tangent_projection(M,vf,normalize); 
M.visualize(f,plt_override); 
add_vectorfield(M,vfp,[],plt_override); 

% cameratoolbar; cameratoolbar('SetCoordSys','none');
add_xyz_axis(); 
end

