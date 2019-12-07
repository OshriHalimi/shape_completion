function [K,H,stable]= visualize_curvature(M,meth,opt)

if ~exist('meth','var'); meth = 'szymon'; end
o.Eps = 0; 
o.smooth_meth = 'none'; 
o.smooth_param = 0; 
if exist('opt','var'); o = mergestruct(o,opt); end

[K,H,stable] = mesh_curvature(M,meth,o); 
ov.clr_map.trunc ='outlier';
% M.plt.clr_map.name = 'half_jet';
ov.clr_bar =1; 
ov.clr_map.name = 'bluegreenred';

M.ezvisualize(K,ov); title(uplw(sprintf('Gaussian Curvature via %s eig smoothing = %s',meth,o.smooth_meth))); 
M.ezvisualize(H,ov); title(uplw(sprintf('Mean Curvature via %s with eig smoothing = %s',meth,o.smooth_meth))); 

% Classification Plot: 
M.plt.trunc_clr_map =0;
hyperbolic = (K<0); 
parabolic = (K==0) & (H~=0); 
elliptic = (K>0); 
planar = (K==0) & (H==0); 
classification = elliptic*1+hyperbolic*2+parabolic*3+planar*4;
classification(1) = 4; classification(end) = 0; % So colorbar would always be full. 
M.ezvisualize(classification,ov); title('PDE classifcation'); 
colorbar('Ticks',0:4,'TickLabels',{'Zeroset','Elliptic','Hyperbolic','Parabolic','Planar'}); 