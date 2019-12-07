function visualize_volume_slices(W, n_clrs)

if nargin<2
    n_clrs = 256;
end

[n,p,q] = size(W);
if islogical(W)
    W = single(W);
end
h = slice(W,p/2,n/2,q/2);

% Touchups 
set(h,'FaceColor','interp','EdgeColor','none');
set(gcf,'color','w');
box on;
set(gca, 'XTick', []);
set(gca, 'YTick', []);
set(gca, 'ZTick', []);
colormap( jet(n_clrs) );
lighting phong;
camlight infinite; 
camproj('perspective');
view(3);
axis tight;
axis equal;
