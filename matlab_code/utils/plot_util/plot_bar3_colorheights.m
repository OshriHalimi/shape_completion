function [] = bar3_colorheights( x, y, Z )
%PLOT_BAR3_COLORHEIGHTS bar3 plot with color coded heights
% x, y : vector of centers of bins
%    Z : matrix of bin heights

b = bar3(y,Z,1)

for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end

axis tight

Xdat=get(b,'XData');
for ii=1:length(Xdat)
    Xdat{ii}=(Xdat{ii}-0.5)*(x(2)-x(1));
    set(b(ii),'XData',Xdat{ii});
end

end

