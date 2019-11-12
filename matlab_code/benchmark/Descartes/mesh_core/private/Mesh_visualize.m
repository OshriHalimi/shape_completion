function [h] = Mesh_visualize(M)
S = M.plt.S;
g = groot;
if isempty(g.Children) || M.plt.new_fig
    fullfig;
elseif M.plt.do_clf ==1
    clf;
end

S.Vertices = M.v; S.Faces = M.f;
% trisurf(M.f,M.v(:,1),M.v(:,2),M.v(:,3),S.FaceVertexCData,'EdgeColor','none')
h = patch(S); title(M.plt.title);% TODO: Think about adding this: 'edgecolor','interp','EdgeAlpha',0.2);
if(M.plt.clr_bar)
    colorbar;
end

%%%%%%%%%%%%%%%% - SETS LIGHT AND DISP ANGLE - %%%%%%%%%%%%%%%%%%%%%%%%%%%
view(M.plt.disp_ang);

if M.plt.light %&& isempty(findall(gcf,'Type','light'))
    % This enables rotation with light change
    c = camlight('headlight'); %lighting('phong');
    set(c,'style','infinite');    % Set style
    b = rotate3d;                 % Create rotate3d-handle
    b.ActionPostCallback = @RotationCallback; % assign callback-function
    b.Enable = 'on';              % no need to click the UI-button
end
if ~isempty(M.plt.post_disp_ang)
    view(M.plt.post_disp_ang);
end
%-------------------------------------------------------------------------%
axis('image'); axis('off');
% cameratoolbar;
colormap(M.plt.clr_map.name);
if M.plt.clr_map.invert
    set(gcf,'Colormap',flipud(get(gcf,'Colormap')));
end
if isfield(S,'FaceVertexCData') && size(S.FaceVertexCData,2)==1
    f = S.FaceVertexCData;
    scale = M.plt.clr_map.scale;
    if isfield(M.plt.clr_map,'axis')
        caxis(M.plt.clr_map.axis)
    else
        switch M.plt.clr_map.trunc
            case 'none'
                caxis([min(f) max(f)+eps]); 
            case 'sym'
                if scale == -1 % Default value
                    scale = 1;
                end
                c = scale*max(abs(f));
                caxis([-c c]);
            case 'outlier'
                if scale == -1
                    scale = 0.95;
                end
                x = sort(f,'descend');
                minc = x(round(length(x)*0.95));
                maxc = x(round(length(x)*(1-scale)));
                caxis([minc maxc]);
        end
    end
end
if M.plt.limits
    xlim([min(M.x()),max(M.x())+eps]); 
    ylim([min(M.y()),max(M.y())+eps]); 
    zlim([min(M.z()),max(M.z())+eps]); 
end
% set(h,'SpecularColorReflectance',0.1,'SpecularExponent',200,'ambientstrength',0.35);
material('shiny'); 
camproj('perspective')
if M.plt.do_clf ~= 2
    drawnow; % So title would always be added
end
end
%-------------------------------------------------------------------------%
%
%-------------------------------------------------------------------------%
function RotationCallback(f,ax)
% c = findall(gcf,'Type','light');
all_axes = findall(f,'type','axes').';
if numel(all_axes)~=2 % Pair Plot
    all_axes = ax.Axes;
end
for ax=all_axes
for i=numel(ax.Children):-1:1
    if isa(ax.Children(i),'matlab.graphics.primitive.Light')
        camlight(ax.Children(i),'headlight');
        break; % Presume only one light.
    end
end
end
end

% MORE COLOR OPTIONS: 
% if strcmp(colorfx, 'wrapping')
%     num = 20;
%     col = rescale(col);
%     col = mod(col*num,2); col(col>1) = 2-col(col>1);
% end
% if strcmp(colorfx, 'equalize')    
%     col = perform_histogram_equalization(col, linspace(0,1,length(col)));
% end
