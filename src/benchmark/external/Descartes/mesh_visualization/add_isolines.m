function [H,V] = add_isolines(Surf,F,V,Col)
% IsoLine    plot isolines of a function on a surface
%    Unlike programs from the CONTOUR-family, IsoLine can handle 
%    arbitrary functions defined on the surface. Further, the 
%    surface can be given as a rectangular or a triangular mesh.
%
%    The call [H,V] = IsoLine(Surf,F,V,Col) is defined as follows:
%
%    Surf is a cell array containing a surface representation. It is
%      * either a triangular mesh {T,[x y z]}, where the rows of T  
%      are indices of vertices forming triangles, and the rows of
%      [x y z] are the coordinates of the vertices,
%      * or a rectangular mesh {X,Y,Z} with matrices of equal size.
%    F are the function values at the vertices, given as a vector 
%      the size of x or as a matrix the size of X, respectively.
%    V determines the values of F to be traced (default V=20).
%      As in CONTOUR, a singleton specifies the number of equidistant 
%      values spread over the range of F, while a vector specifies 
%      specific values. Use V = [v v] to draw a single line at level v.
%    Col is a character defining the color of isolines (default Col='k').
%
%    H is a vector containing the grahics handles of the isolines.
%    V is a vector containing the traced function values, what may be 
%      useful if V was given as a number.
%
% See also: IsoLineDemo, contour, contour3, delaunay, mesh
% -------------------------------------------------------------------------
% Author:  Ulrich Reif
% Date:    March 12, 2015
% Version: Tested with MATLAB R2014b and R2012b
% -------------------------------------------------------------------------
if isa(Surf,'Mesh')
    Surf = {Surf.f,Surf.v};
end

% Preprocess input --------------------------------------------------------
if length(Surf)==3                % convert mesh to triangulation
  P = [Surf{1}(:) Surf{2}(:) Surf{3}(:)];
  Surf{1}(end,:) = 1i;
  Surf{1}(:,end) = 1i;
  i = find(~imag(Surf{1}(:)));
  n = size(Surf{1},1);
  T = [i i+1 i+n; i+1 i+n+1 i+n];
else
  T = Surf{1};
  P = Surf{2};
end
f = F(T(:));
if nargin==2
  V = linspace(min(f),max(f),22);
  V = V(2:end-1);
elseif numel(V)==1
  V = linspace(min(f),max(f),V+2);
  V = V(2:end-1);
end
if nargin<4
  Col = 'k';
end
H = NaN + V(:);
q = [1:3 1:3];
% -------------------------------------------------------------------------


% Loop over iso-values ----------------------------------------------------
for k = 1:numel(V)
  R = {[],[]};
  G = F(T) - V(k);   
  C = 1./(1-G./G(:,[2 3 1]));
  f = unique(T(~isfinite(C))); % remove degeneracies by random perturbation
  F(f) = F(f).*(1+1e-12*rand(size(F(f)))) + 1e-12*rand(size(F(f)));
  G = F(T) - V(k);
  C = 1./(1-G./G(:,[2 3 1]));
  C(C<0|C>1) = -1;
  % process active triangles
  for i = 1:3
    f = any(C>=0,2) & C(:,i)<0;
    for j = i+1:i+2
      w = C(f,q([j j j]));
      R{j-i} = [R{j-i}; w.*P(T(f,q(j)),:)+(1-w).*P(T(f,q(j+1)),:)];
    end
  end
  % define isoline
  for i = 1:3
    X{i} = [R{1}(:,i) R{2}(:,i) nan+R{1}(:,i)]';
    X{i} = X{i}(:)';
  end
  % plot isoline
  if ~isempty(R{1})
    hold on
    H(k) = plot3(X{1},X{2},X{3},Col,'LineWidth',1);
  end
end
% -------------------------------------------------------------------------


% Render with 'zbuffer' for best results ----------------------------------
set(gcf,'Renderer','zbuffer')
% -------------------------------------------------------------------------
