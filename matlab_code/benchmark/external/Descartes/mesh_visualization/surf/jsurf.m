function h = jsurf(varargin)
%
% JSURF plots a 3D surface onto a second surface
%
%   JSURF is similar in almost all respects to the standard SURF plot except
%   that it admits a second matrix T of z-coordinates of size [m,n] that 
%   is used to plot over the base surface described by coordinates (x,y,z). 
%
%   JSURF(x,y,z,T) plots the 3D surface T onto the surface described by
%   (x,y,z). JSURF(x,y,z,-T) plots on the underside of the surface.
%   The colour scale is determined by the range of T. 
%
%   JSURF(x,y,z,T,C) uses a colour scale determined by C. Input x, y, z,
%   T and C must all have dimension [m,n]. 
%
%   If two such matrices are passed to the function as in JSURF(z,T), it 
%   interprets the first as the base surface, and the second is plotted  
%   onto the first, whereas if three matrices are passed to the function  
%   as in JSURF(z,T,C), the third matrix of values C is interpreted as the 
%   associated colour map. In these cases, x and y are both [1:m,1:n] and 
%   same size as T.
%
%   JSURF(...,'PropertyName','PropertyValue',...) sets the surface 
%   property values for the object.
%
%   h = JSURF(...) returns the handle to the plot object.
%
%   AXIS, CAXIS, COLORMAP, HOLD, SHADING and VIEW set figure, axes, and 
%   surface properties which affect the display of the surface as in SURF.
%
%   JSURF(...,'v6') creates a surface object instead of a surface plot
%   object for compatibility with MATLAB 6.5 and earlier, only if the
%   internal function 'axescheck.m' is available to the user.
%
%   See also SURF, CYL3D and SPHERE3D    

%   JM De Freitas
%   QinetiQ Ltd, Winfrith Technology Centre
%   Winfrith, Dorchester DT2 8XJ. UK.
%   Email: jdefreitas@qinetiq.com
%   Date: 17th October 2005

args = varargin;
[ax,args,nargs] = axescheck(args{:});

for i = 1:nargs 
    if (ischar(args{i})) nargs = i - 1; break; end 
end
if nargs <= 1
    error('JSURF error: too few input arguments.');
end
if nargs > 5
    error('JSURF error: too many input numerical arguments.');
end

for i = 1:nargs [r(i),c(i)] = size(args{i}(:,:)); end
[rx,cx] = size(args{1}(:,:));
dim(1:nargs-1) = false;
for i = 1:nargs-1 
    if (r(i) == r(i+1))&(c(i) == c(i+1)) dim(i) = true; end
end

if (nargs == 2)&(all(dim(:) == true))&~ischar(args)
    z = args{1}(:,:); T = args{2}(:,:); C = T; 
    [x,y] = meshgrid(1:rx,1:cx); 
elseif (nargs == 2)& any(dim(:) ~= true)
    error('JSURF error: z must be the same size as T.');
end

if (nargs == 3)&(all(dim(:) == true))&~ischar(args)
    z = args{1}(:,:); T = args{2}(:,:); C = args{3}(:,:);  
    [x,y] = meshgrid(1:rx,1:cx);
elseif (nargs == 3)& any(dim(:) ~= true)
    error('JSURF error: z and T must be the same size as C.');
end

if (nargs == 4)&(all(dim(:) == true))&~ischar(args)
    x = args{1}(:,:);y = args{2}(:,:);z = args{3}(:,:);
    T = args{4}(:,:); C = T;
elseif (nargs == 4)&any(dim(:) ~= true)
    error('JSURF error: x, y and z must be the same size as T.');
end

if (nargs == 5)&(all(dim(:) == true))&~ischar(args)
    x = args{1}(:,:);y = args{2}(:,:);z = args{3}(:,:); 
    T = args{4}(:,:);C = args{5}(:,:); 
elseif (nargs == 5)&any(dim(:) ~= true)
    error('JSURF error: x, y, z and T must be the same size as C.');
end

if isempty(ax) 
  ax = newplot;
end

[nx,ny,nz] = surfnorm(x,y,z); 
x = T.*nx + x;
y = T.*ny + y;
z = T.*nz + z;
ha = surf(x,y,z,C,'parent',ax,args{nargs+1:end});

if nargout == 1
    h = ha;
else
    h = [];
end
return