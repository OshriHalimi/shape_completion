function add_vectorfield(M,vf,opt,plt_override)

o.pick = 0;
o.normalize = 1; 
if exist('opt','var') && ~isempty(opt); o = mergestruct(o,opt); end
if exist('plt_override','var'); M.plt = mergestruct(M.plt,plt_override); end

% Allow for VF Flexibility
if size(vf,2) == 1; vf = reshape(vf,M.Nf,3); end
% Handle the opts: 
if o.normalize; vf= normr(vf); end

% Decide on arrow bases: 
if size(vf,1) == M.Nf
    M = M.add_face_normals(); 
    cc = M.fc;
elseif size(vf,1) == M.Nv
    cc = M.v; 
else
    error('Invalid size'); 
end
    
% Sparsify
[vf,ids] = sparsify(vf,o.pick); 
cc = cc(ids,:); 

hold on; 
h = quiver3(cc(:,1),cc(:,2),cc(:,3),vf(:,1),vf(:,2),vf(:,3),0); 
hold off;

% This is due to the AutoScale feature quiver3 + the need to rescale:
set(h,'UData',M.plt.normal_scaling*get(h,'UData'),...
      'VData',M.plt.normal_scaling*get(h,'VData'),...
      'WData',M.plt.normal_scaling*get(h,'WData'));
if ~isempty(M.plt.normal_clr)
    h.Color = M.plt.normal_clr;
end

end