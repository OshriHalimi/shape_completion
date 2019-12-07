function hh = quiver3_elips(x,y,z,u,v,w,varargin) 
%VECTOR3 3-D vector plot.
% VECTOR3(X,Y,Z,U,V,W) plots  vectors as colored ellipsoids with components
% (u,v,w) at the points (x,y,z).  The matrices X,Y,Z,U,V,W must all be the
% same size and contain the corresponding position and vector components.
% VECTOR3 automatically scales the ellipsoids to fit.
%
% Parameters
% Resolution: # of vertices along the longitudinal axis of the ellipsoids.
% Scale:      Length of the ellipsoids in the same units as x, y, z.
% Colormap:   Colormap used on all of the ellipsoids.
%
%% Example
% % Calculate the surface normals of a function.
% x = linspace(-pi,pi,20);
% [X,Y] = meshgrid(x);
% Z = sin(X).*sin(Y);
% [U,V,W] = surfnorm(X,Y,Z);
% 
% % Scale normal vectors to the surface's curvature
% C = del2(Z);
% C = C./max(abs(C(:)));
% U = U.*(sign(C)+C);
% V = V.*(sign(C)+C);
% W = W.*(sign(C)+C);
% 
% % Plot the vector field
% hh = vector3(X,Y,Z,U,V,W,...
%   'Resolution', 20,...
%   'Scale', 0.75*abs(Y(2)-Y(1)),...
%   'Colormap', colormap(bone(40)));
%
%%
% Copyright 2014 The MathWorks, Inc.
%
%% Code
% Parse input
p = inputParser;
defaultEllipsoidResolution = 20; % number of vertices along longitude
defaultEllipsoidScale = 0.5*abs(y(2)-y(1)); % length of the ellipsoids
defaultCMap = hot(2*defaultEllipsoidResolution);

addRequired(p,'x',@(x)(isnumeric(x) && isreal(x)));
addRequired(p,'y',@(x)(isnumeric(x) && isreal(x)));
addRequired(p,'z',@(x)(isnumeric(x) && isreal(x)));
addRequired(p,'u',@(x)(isnumeric(x) && isreal(x)));
addRequired(p,'v',@(x)(isnumeric(x) && isreal(x)));
addRequired(p,'w',@(x)(isnumeric(x) && isreal(x)));

addParameter(p,'Resolution',defaultEllipsoidResolution,@isnumeric);
addParameter(p,'Scale'     ,defaultEllipsoidScale     ,@isnumeric);
addParameter(p,'Colormap'  ,defaultCMap               ,@isnumeric);

parse(p,x,y,z,u,v,w,varargin{:});
ellipsoidResolution = p.Results.Resolution;
ellipsoidScale = p.Results.Scale;
cMap = p.Results.Colormap;
fullfig;
colormap(cMap); 



% Get norms and unit vectors
vec     = [u(:),v(:),w(:)];
vecNorm = reshape(sqrt(u.^2 + v.^2 + w.^2),[],1);
vec     = bsxfun(@rdivide,vec,vecNorm);
vecNorm = vecNorm ./ max(vecNorm);

% Euler angles to orient ellipsoids
beta = acos(vec(:,3));
alfa = atan2(vec(:,1),-vec(:,2));

% Prototype ellipsoid used to build all other ellipsoids
[ex,ey,ez] = ellipsoid(0,0,0,ellipsoidScale*2/(1+sqrt(5)),ellipsoidScale*2/(1+sqrt(5)),ellipsoidScale,ellipsoidResolution);
proto = surf(ex,ey,ez);
proto.EdgeColor    = 'none';
proto.FaceColor    = 'interp';
proto.FaceLighting = 'none';
proto.CDataMapping = 'direct';

% Set up some graphics options
axis equal;
axis vis3d;
set(gcf,'Renderer','opengl');

% Preallocate graphics object vectors
numgobjects = numel(x);
E = gobjects(numgobjects,1); % ellipses
T = gobjects(numgobjects,1); % transforms
for i = 1:numgobjects
  T(i) = hgtransform;
  % Create a copy of the prototype ellipsoid, parent it to an hgtransform
  E(i) = copyobj(proto,T(i));
  if vecNorm(i) < eps % Don't show zero vectors at all, but let them exist so you can see them if you want.
    E(i).Visible = 'off';
  else
    % Perform rotations then translation
    M = makehgtform('translate',[x(i) y(i) z(i)],'zrotate',alfa(i),'xrotate',beta(i));
    T(i).Matrix = M; % this line performs the operation defined above.
  end
  % Scale colors based on relative magnitudes of the vectors vecNorm(i)
  E(i).CData = repmat(linspace(1,vecNorm(i)*length(cMap),ellipsoidResolution+1).',1,ellipsoidResolution+1);
end

% Delete prototype ellipsoid
proto.delete;

% Return an hggroup object. The hierarch is hh > hgtransform > ellipsoid
hh = hggroup;
set(T,'Parent',hh);
end