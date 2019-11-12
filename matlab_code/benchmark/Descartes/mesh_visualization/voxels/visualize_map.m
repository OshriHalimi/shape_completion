function visualize_map(C,ax)
if ~exist('ax','var'); ax = 1; end
if size(C,3) == 1
    display_map(C,ax); 
else
    display_tensor(C,ax);
end
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function display_map(C,ax)
fullfig; imagesc(C);axis equal;  
xlim([1,size(C,2)]);  ylim([1,size(C,1)]);  
if ~ax
    axis('off'); 
end
colormap('jet');
cameratoolbar;
end
function display_tensor(C,ax)
% Usage: m = rand(7,7,7); m(2,3,4) = 1; display_map(m); 
[nx,ny,nz] = size(C);

% Generate the squared patches: 
[yrng_org, xrng_org, zrng_org] = meshgrid(1:ny,1:nx,1:nz);
xrng(1,:,:,:) = xrng_org-0.5;
xrng(2,:,:,:) = xrng_org+0.5;
yrng(1,:,:,:) = yrng_org-0.5;
yrng(2,:,:,:) = yrng_org+0.5;
zrng(1,:,:,:) = zrng_org-0.5;
zrng(2,:,:,:) = zrng_org+0.5;
tbl = [1 1 1 2;1 1 2 2; 1 2 2 2; 1 2 1 2];
xtbl = tbl(:,[3 3 4 1 3 3]);
ytbl = tbl(:,[2 2 3 3 4 1]);
ztbl = tbl(:,[4 1 2 2 2 2]);
x_tot = zeros(4,6,nx,ny,nz);y_tot = x_tot; z_tot = x_tot;
for vertex = 1:4
  for face = 1:6
      x_tot(vertex,face,:,:,:) = xrng(xtbl(vertex,face),:,:,:);
      y_tot(vertex,face,:,:,:) = yrng(ytbl(vertex,face),:,:,:);
      z_tot(vertex,face,:,:,:) = zrng(ztbl(vertex,face),:,:,:);
  end
end
x_tot = reshape(x_tot, 4, nx*ny*nz*6);
y_tot = reshape(y_tot, 4, nx*ny*nz*6);
z_tot = reshape(z_tot, 4, nx*ny*nz*6);

% Set the alpha mat and draw 
alphamat = (C(:)*ones(1,6))';
fullfig;axis('equal');cameratoolbar;
h = patch(x_tot,y_tot,z_tot,alphamat(:)); colormap('jet'); 
set(h,'FaceVertexAlphaData',alphamat(:)); set(h, 'FaceAlpha', 'flat'); set(h, 'EdgeAlpha', 0);

axis([0.5 nx+0.5 0.5 ny+0.5 0.5 nz+0.5])
view([41,28]);
if ax
    xlabel('x'); ylabel('y'); zlabel('z');
else
    axis('off'); 
end
end