function [seg] = visualize_segmentation_fast(M,seg,opt,oplt)

o.disp_func = @ezvisualize; o.meth_name = 'Segmentation'; 
if exist('opt','var') && ~isempty(opt); o = mergestruct(o,opt,1);end

n_clrs = sum(unique(seg)>0); n_uncolored = sum(seg <= 0);
prec = 100*(n_uncolored/numel(seg));
if numel(seg)==M.Nf
    M.plt.title = usprintf('%s on Faces with %d colors [%g %% Uncolored Faces]',o.meth_name, n_clrs,prec);
else
    M.plt.title = usprintf('%s on Vertices with %d colors [%g %% Uncolored Vertices]',o.meth_name,n_clrs,prec);
end
ov = struct(); 
if numel(seg)==M.Nf; ov.S.LineStyle = 'none'; end
if exist('oplt','var') && ~isempty(oplt); ov = mergestruct(ov,oplt,1);end

% To make the segementation not reliant on the number, we randomize the
% segmentation order
perm = randperm(n_clrs); 
seg=perm(seg);
if size(seg,2) ~= 1
    seg = seg.';
end
o.disp_func(M,seg,ov); 
end