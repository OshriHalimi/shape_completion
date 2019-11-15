function [seg] = visualize_segmentation(M,seg,opt,oplt)

n_clrs = sum(unique(seg)>0);
n_uncolored = sum(seg <= 0);

o.seg_clrs = num2cell(flipud(hsv(n_clrs)),2); 
o.init_clr = 'w'; o.clr_extend = 'rand'; 
o.disp_func = @ezvisualize; 
o.meth_name = 'Segmentation'; 
if exist('opt','var') && ~isempty(opt); o = mergestruct(o,opt,1);end

clrs = uniclr(o.init_clr,numel(seg));
for i=1:n_clrs
    if i> numel(o.seg_clrs)
        clr = o.clr_extend;
    else
        clr = o.seg_clrs{i};
    end
    ids = (seg == i);
    clrs(ids,:) = uniclr(clr,sum(ids));
end
prec = 100*(n_uncolored/numel(seg));
if numel(seg)==M.Nf
    M.plt.title = usprintf('%s on Faces with %d colors [%g %% Uncolored Faces]',o.meth_name, n_clrs,prec);
else
    M.plt.title = usprintf('%s on Vertices with %d colors [%g %% Uncolored Vertices]',o.meth_name,n_clrs,prec);
end
ov.clr_bar = 0;
if numel(seg)==M.Nf; ov.S.LineStyle = 'none'; end
if exist('oplt','var') && ~isempty(oplt); ov = mergestruct(ov,oplt,1);end
o.disp_func(M,clrs,ov); 
end