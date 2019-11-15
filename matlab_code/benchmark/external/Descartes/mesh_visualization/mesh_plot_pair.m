function mesh_plot_pair(M1,M2,C1,C2,oplt1,oplt2)

% Handle Default Params: 
if ~exist('C1','var'); C1 = []; end
if ~exist('C2','var'); C2 = []; end

o1.disp_func = @ezvisualize; o1.title = ''; o1.clr_bar = 0; o1.new_fig = 1;
if exist('oplt1','var'); o1 = mergestruct(o1,oplt1,2);end
if ~exist('oplt2','var') 
    o2 = o1;
else
    o2.disp_func = @ezvisualize; o2.title = ''; o2.clr_bar = 0;  o1.new_fig = 0;
    o2 = mergestruct(o2,oplt2,2);
end

g = groot;
if isempty(g.Children) || o1.new_fig || o2.new_fig
    fullfig;
end
o1.do_clf = 2; o1.new_fig = 0; % Can't have these on - we are plotting two elements
o2.do_clf = 2; o2.new_fig = 0;
clf; % For possible animations 

ax1 = subplot_tight(1,2,1,[0.03,0.03]);
o1.disp_func(M1,C1,o1);
ax2 = subplot_tight(1,2,2,[0.03,0.03]);
o2.disp_func(M2,C2,o2);
Link = linkprop([ax1, ax2],{'CameraUpVector', 'CameraPosition', 'CameraTarget'});
setappdata(gcf, 'StoreTheLink', Link);
drawnow; % Draw them together