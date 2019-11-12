function IsoLineDemo
%% EXAMPLE 1: Isolines of f(x,y,z) = xy+20z on a triangular mesh -----------
figure(1), clf reset                       
set(1,'Name','Isolines of f(x,y,z) = xy+20z on a triangular mesh')
load trimesh3d                           % sample triangulation
F = x.*y+ 20*z;                          % function values 
trisurf(tri,x,y,z,F), shading interp     % plot surface, colored by F
Surf = {tri,[x y z]};                    % surface as cell array
[~,V]= add_isolines(Surf,F);                  % plot isolines        
camlight, axis equal                     % figure settings
view(-35,-27)
%
% Basically, that's all - but let's add some gimmicks ...
%
% Draw a colorbar, labelled with the iso-values
J = (.5+hsv(21))/1.5;                    % colormap, slightly whitened
J = J(floor((3:65)/3),:);                % triple entries
colormap(J);                             % apply J
c = colorbar('YTick',round(V));
%
% This push-button activates a spin of colors.
uicontrol('Style','ToggleButton',...       
  'Position',[10 10 65 25],'string','PUSH','FontWeight','bold',...
  'CallBack',{@spin,J,V,c},'ToolTipString','spin colors');    
% -------------------------------------------------------------------------


%% EXAMPLE 2: Intersection of a cylinder and some concentric balls ---------
figure(2), clf reset                 
set(2,'Name','Intersection of a cylinder and some balls')
[x,y,z] = cylinder(ones(1,30),90);       % unit cylinder
z = 2*z-1;                               % stretch and shift
R = sqrt(x.^2+(y+1).^2+z.^2);            % distance from point [0,1,0]
mesh(x,y,z,'EdgeColor','c')              % plot cylinder
Surf = {x,y,z};                          % surface as cell array
T = (1:16)/8;                            % values to be tracked
H = add_isolines(Surf,R,T,'b');               % plot blue isolines
set(H(8),'Color','r')                    % red isoline at level R=1
axis equal;                    % figure settings
% -------------------------------------------------------------------------


%% EXAMPLE 3: Subdivide peaks graph into elliptic and hyperbolic parts -----
figure(3), clf reset                     
set(3,'Name','Elliptic and hyperbolic parts of the peaks function')
[x,y,z] = peaks(181);                    % peaks function
i = 2:length(x)-1;                     
zxx = z(i,i+1) - 2*z(i,i) + z(i,i-1);    % scaled second derivatives
zxy = (z(i+1,i+1) - z(i-1,i+1) - z(i+1,i-1) + z(i-1,i-1))/4;
zyy = z(i+1,i) - 2*z(i,i) + z(i-1,i);
D = nan*x;
D(i,i) = zxx.*zyy-zxy.^2;                % determinant of Hessian
D = 3*D/max(D(:));                       % normalize range
surf(x,y,z,D+(D>0)-(D<0))                % graph, colored by augmented D
shading interp
Surf = {x,y,z};                          % surface as cell array
h = add_isolines(Surf,D,[0 0]);               % zero contour of D
set(h,'LineWidth',2)                     % fat line
i = 1:30:181;
mesh(x(i,i),y(i,i),0*z(i,i)-8,'FaceColor','none')   % plane underneath
Surf = {x,y,0*z-8};                      % plane underneath as cell array
h = add_isolines(Surf,D,[0 0],'r');           % planar zero contour of H
set(h,'LineWidth',2)
caxis([-3 3]), camlight                  % figure settings
zoom(1.6), axis tight
% ------------------------------------------------------------------------

% Auxiliary function for Example 1 ----------------------------------------
function [] = spin(u,~,J,V,c)
set(u,'String','WAIT')
for i=1:size(J,1)
  tic
  J = J([2:end 1],:);                    % spin colormap
  colormap(J)
  set(c,'YTick',round(V))                % reset tick labels (for R2012b)
  drawnow
  while toc<.1,end                       % slow down, if necessary
end
set(u,'String','PUSH','Value',0)
% -------------------------------------------------------------------------

