function V = SurfaceSmooth(Vertices, Faces, VoxSize, DisplTol, IterTol, Freedom, Verbose)
  % Smooth closed triangulated surface to remove "voxel" artefacts.
  %
  % V = SurfaceSmooth(Vertices, Faces, VoxSize, DisplTol, IterTol, Freedom, Verbose)
  % V = SurfaceSmooth(Patch, [], VoxSize, DisplTol, IterTol, Freedom, Verbose)
  %
  % Smooth a triangulated surface, trying to optimize getting rid of blocky
  % voxel segmentation artefacts but still respect initial segmentation.
  % This is achieved by restricting vertex displacement along the surface
  % normal to half the voxel size, and by compensating normal displacement
  % of a voxel by an opposite distributed shift in its neighbors.  That
  % way, the total normal displacement is approximately zero (before
  % potential restrictions are applied).
  %
  % Tangential motion is currently left unrestricted, which means the mesh
  % will readjust over many iterations, much more than necessary to obtain
  % a smoothed surface.  On the other hand, this produces a more uniform
  % triangulation, which may be desirable in some cases, e.g. after a
  % reducepatch operation.  This tangential motion may also make the normal
  % restriction a bit less accurate.  This all depends on how irregular the
  % mesh was to start with.  To avoid long running times and some of the
  % tangential deformation, DisplTol and IterTol can be used to limit the
  % number of iterations.
  %
  % To separate these two effects (normal smoothing and tangential mesh
  % uniformization), the function can be run to achieve each type of motion
  % separately, by setting Freedom to the appropriate value (see below).
  % Even if both effects are desired, but precision in the normal
  % displacement restriction is preferred over running time, I would
  % suggest running twice (norm., tang.) or three times (tang., norm.,
  % tang.), but only once in the normal direction.  Note that tangential
  % motion is not perfect and may cause a small amount of smoothing as
  % well.
  %
  % Input variables:
  %  Vertices [nV, 3]: Point 3d coordinates.
  %  Faces [nF, 3]: Triangles, i.e. 3 point indices.
  %  Patch: Instead of Vertices and Faces, a single structure can be given
  %    with fields 'vertices' and 'faces'.  In this case, leave Faces empty
  %    [].
  %  VoxSize (default inf): Length of voxels, this determines the amount of
  %    smoothing.  For a voxel size of 1, vertices are allowed to move only
  %    0.4 units in the surface normal direction.  This is somewhat optimal
  %    for getting rid of artefacts: it allows steps to become flat even at
  %    shallow angles and a single voxel cube would be transformed to a
  %    sphere of identical voxel volume.
  %  DisplTol (default 0.01*VoxSize): Once the maximum displacement of 
  %    vertices is less than this distance, the algorithm stops.  If two
  %    values are given, e.g. [0.01, 0.01], the second value is compared to
  %    normal displacement only. The first limit encountered stops
  %    iterating.  This allows stopping earlier if only smoothing is
  %    desired and not mesh uniformity.
  %  IterTol (default 100): If the algorithm did not converge, it will stop
  %    after this many iterations.
  %  Freedom (default 2): Indicate which motion is allowed by the
  %    algorithm with an integer value: 0 for (restricted) normal
  %    smoothing, 1 for (unrestricted) tangential motion to get a more
  %    uniform triangulation, or 2 for both at the same time.
  %  Verbose (default false): If true, writes initial and final volumes and 
  %    areas on the command line.  Also gives the number of iterations and
  %    final displacement when the algorithm converged.  (A warning is
  %    always given if convergence was not obtained in IterTol iterations.)
  %
  % Output: Modified voxel coordinates [nV, 3].
  %
  % Written by Marc Lalancette, Toronto, Canada, 2014-02-04
  % Volume calculation from divergence theorem idea: 
  %  http://www.mathworks.com/matlabcentral/fileexchange/26982-volume-of-a-surface-triangulation
  
  % Note: Although this seems to work relatively well, it is still very new
  % and not fully tested.  Despite the description above which is what was
  % intended, the algorithm had the tendency to drive growing oscillations
  % (from iteration to iteration) on the surface.  Thus a basic damping
  % mechanism was added: I simply multiply each movement by a fraction that
  % seems to avoid oscillations and still converge rapidly enough.
  
  if ~exist('Freedom', 'var') || isempty(Freedom)
    Freedom = 2; % 0=norm, 1=tang, 2=both.
  end
  
  % Attempt at damping oscillations. Reduce any movement by a certain
  % fraction. (Multiply movements by this factor.)
  DampingFactor = 0.91;
  
  if nargin < 2 || isempty(Faces)
    if isfield(Vertices, 'faces')
      Faces = Vertices.faces;
    elseif isfield(Vertices, 'Faces')
      Faces = Vertices.Faces;
    else
      error('Faces required as second input or "faces" field of first input.');
    end
    if isfield(Vertices, 'vertices')
      Vertices = Vertices.vertices;
    elseif isfield(Vertices, 'Vertices')
      Vertices = Vertices.Vertices;
    else
      error('Patch.vertices field required when second input is empty.');
    end
  end
  if ~exist('VoxSize', 'var') || isempty(VoxSize)
    VoxSize = inf;
    if ~exist('IterTol', 'var') || isempty(IterTol)
      error(['Unrestricted smoothing (no VoxSize) would lead to a sphere of similar volume, ', ...
        'unless limited by the number of iterations.']);
    end
  end
  if ~exist('DisplTol', 'var') || isempty(DisplTol)
    DisplTol = 0.01 * VoxSize;
  end
  if numel(DisplTol) == 1
    % Only stop when total displacement reaches the limit.
    DisplTol = [DisplTol, 0];
  end
  if ~exist('IterTol', 'var') || isempty(IterTol)
    IterTol = 100;
  end
  if ~exist('Verbose', 'var') || isempty(Verbose)
    Verbose = false;
  end
  
  % Verify surface is a triangulation.
  if size(Faces, 2) > 3
    error('SurfaceSmooth only works with a triangulated surface.');
  end
  
  % Optimal allowed normal displacement, in units of voxel side length.
  % Based on turning a single voxel into a sphere of same volume: max
  % needed displacement is in corner:
  %  sqrt(3)/2 - 1/(4/3*pi)^(1/3) = 0.2457
  % In middle of face it is rather:
  %  1/(4/3*pi)^(1/3) - 1/2 = 0.1204
  % Based on very gentle sloped staircase, it would be 0.5, but for 45
  % degree steps, we only need cos(pi/4)/2 = 0.3536.  So something along
  % those lines seems like a good compromize.  For now try to make steps
  % completely disappear.
  MaxNormDispl = 0.5 * VoxSize;
  %   MaxDispl = 2 * VoxSize; % To avoid large scale slow flows tangentially, which could distort.
  
  nV = size(Vertices, 1);
  %   nF = size(Faces, 1);
  
  % Remove duplicate faces.  Not necessary considering we have to use
  % unique on the edges later anyway.
  %   Faces = unique(Faces, 'rows');
  
  if Verbose
    [~, ~, FNdA, FdA] = VertexNormals(Vertices);
    FaceCentroidZ = ( Vertices(Faces(:, 1), 3) + ...
      Vertices(Faces(:, 2), 3) + Vertices(Faces(:, 3), 3) ) /3;
    Pre.Volume = FaceCentroidZ' * FNdA(:, 3);
    Pre.Area = sum(FdA);
    fprintf('Total enclosed volume before smoothing: %g\n', Pre.Volume);
    fprintf('Total area before smoothing: %g\n', Pre.Area);
  end
  
  % Calculate connectivity matrix.
  
  % Logical matrix would be huge, so use sparse. However tests in R2011b
  % indicate that using logical sparse indexing is sometimes slightly
  % faster (possibly when using linear indexing) but sometimes noticeably
  % slower.  Seems here using a cell array is better.
  
  % This expression works when each edge is found once in each direction,
  % i.e. as long as all normals are consistently pointing in (or out).
  %   C = sparse(Faces(:), [Faces(:, 2); Faces(:, 3); Faces(:, 1)], true);
  % Seems users had patches that didn't satisfy this restriction, or had
  % duplicate faces or had possibly intersecting surfaces with 3 faces
  % sharing an edge.
  Edges = unique([Faces(:), [Faces(:, 2); Faces(:, 3); Faces(:, 1)]], 'rows');
  C = sparse(Edges(:, 1), Edges(:, 2), true);
  C = C | C';
  CCell = cell(nV, 1);
  for v = 1:nV
    CCell{v} = find(C(:, v));
  end
  clear C
  % Number of connected neighbors at each vertex.
  %   nC = full(sum(C, 1));
  
  V = Vertices;
  W = V;
  LastMaxDispl = [inf, inf];
  Iter = 0;
  NormDispl = zeros(nV, 1);
  while LastMaxDispl(1) > DisplTol(1) && LastMaxDispl(2) > DisplTol(2) && ...
      Iter < IterTol
    Iter = Iter + 1;
    [N, VdA] = VertexNormals(V);
    %     W = zeros(nV, 3); % Changed for damping
    VWeighted = VdA * [1, 1, 1] .* V;
    
    % Moving step.  (This is slow.)
    switch Freedom
      case 2 % Both.
        for v = 1:nV
          % Neighborhood average.  Improved to weigh by area element to avoid
          % tangential deformation based on number of neighbors (e.g. shrinking
          % towards vertices with fewer neighbors).
          NeighdA = sum(VdA(CCell{v}));
          a = sum(VWeighted(CCell{v}, :) / NeighdA, 1); % / nC(v);
          % Displacement along normal, divided by number of points that will be
          % shifted inversely.  More accurate using weighing by area element:
          % volume corresponding to this point normal movement, distributed
          % among neighbor area.
          d = (a - V(v, :)) * N(v, :)' / (NeighdA/VdA(v) + 1); % / (nC(v) + 1);
          % Central point is moved to average of neighbors, but shifted back a
          % bit as they all will be.
          W(v, :) = W(v, :) + DampingFactor * ( a - V(v, :) - d * N(v, :) );
          % Neighbors are shifted a bit too along their own normals, such that
          % the total change in volume (normal displacement times surface area)
          % is close to zero.
          W(CCell{v}, :) = W(CCell{v}, :) - DampingFactor * d * N(CCell{v}, :);
        end
      case 0 % Normal motion only.
        for v = 1:nV
          NeighdA = sum(VdA(CCell{v}));
          a = sum(VWeighted(CCell{v}, :) / NeighdA, 1); % / nC(v);
          d = (a - V(v, :)) * N(v, :)' / (NeighdA/VdA(v) + 1); % / (nC(v) + 1);
          W(v, :) = W(v, :) + DampingFactor * NeighdA/VdA(v) * d * N(v, :);
          W(CCell{v}, :) = W(CCell{v}, :) - DampingFactor * d * N(CCell{v}, :);
        end
      case 1 % Tangential motion only.  Unrestricted.
        for v = 1:nV
          NeighdA = sum(VdA(CCell{v}));
          a = sum(VWeighted(CCell{v}, :) / NeighdA, 1); % / nC(v);
          d = (a - V(v, :)) * N(v, :)'; % / (nC(v) + 1);
          W(v, :) = W(v, :) + DampingFactor * ( (a - V(v, :)) - d * N(v, :) );
          % No compensation among neighbors.
        end
      otherwise
        error('Unrecognized Freedom parameter. Should be 0, 1 or 2.');
    end
    % Restricting step.
    % Displacements along normals (N at last positions, V), added to
    % previous normal displacement since we want to restrict total normal
    % displacement.
    D = NormDispl + dot((W - V), N, 2);
    % New restricted total normal displacement.
    NormDispl = sign(D) .* min(abs(D), MaxNormDispl);
    % Amounts to move back if greater than allowed.
    D = D - NormDispl;
    Where = abs(D) > DisplTol(1) * 1e-6; % > 0, but ignore precision errors.
    % Fix.
    if any(Where)
      W(Where, :) = W(Where, :) - [D(Where), D(Where), D(Where)] .* N(Where, :);
    end
    % New restriction on tangential displacement. [Not implemented.]
    %     MaxDispl
    
    
    if Verbose > 1
      [LastMaxDispl(1), iMax(1)] = max(sqrt( sum((W - V).^2, 2)) );
      [LastMaxDispl(2), iMax(2)] = max(abs(dot(W - V, N, 2)));
      TangDisplVec = CrossProduct(W - V, N);
      [LastMaxDispl(3), iMax(3)] = max(sqrt(TangDisplVec(:,1).^2 + TangDisplVec(:,2).^2 + TangDisplVec(:,3).^2));
      fprintf('Iter %d: max displ %1.4g at vox %d; norm %1.4g (vox %d); tang %1.4g (vox %d)\n', ...
        Iter, LastMaxDispl(1), iMax(1), ...
        sign((W(iMax(2),:) - V(iMax(2),:)) * N(iMax(2),:)')*LastMaxDispl(2), iMax(2), ...
        sign(TangDisplVec(iMax(3), 1))*LastMaxDispl(3), iMax(3));
      % Signs are to see if these are oscillations or translations.
    else
      LastMaxDispl(1) = sqrt( max(sum((W - V).^2, 2)) );
      LastMaxDispl(2) = max(dot(W - V, N, 2));
    end
    V = W;
  end
  
  if Iter >= IterTol
    warning('SurfaceSmooth did not converge within %d iterations. \nLast max point displacement = %f', ...
      IterTol, LastMaxDispl(1));
  elseif Verbose
    fprintf('SurfaceSmooth converged in %d iterations. \nLast max point displacement = %f\n', ...
      Iter, LastMaxDispl(1));
  end
  if Verbose && IterTol > 0
    [~, ~, FNdA, FdA] = VertexNormals(V);
    FaceCentroidZ = ( V(Faces(:, 1), 3) + ...
      V(Faces(:, 2), 3) + V(Faces(:, 3), 3) ) /3;
    Post.Volume = FaceCentroidZ' * FNdA(:, 3);
    Post.Area = sum(FdA);
    fprintf('Total enclosed volume after smoothing: %g\n', Post.Volume);
    fprintf('Relative volume change: %g %%\n', ...
      100 * (Post.Volume - Pre.Volume)/Pre.Volume);
    fprintf('Total area after smoothing: %g\n', Post.Area);
    fprintf('Relative area change: %g %%\n', ...
      100 * (Post.Area - Pre.Area)/Pre.Area);
  end
  
  
  
  
  % ----------------------------------------------------------------------
  % Calculate dA normal vectors to each vertex.
  function [N, VdA, FNdA, FdA] = VertexNormals(V)
    N = zeros(nV, 3);
    % Get face normal vectors with length the size of the face area.
    FNdA = CrossProduct( (V(Faces(:, 2), :) - V(Faces(:, 1), :)), ...
      (V(Faces(:, 3), :) - V(Faces(:, 2), :)) ) / 2;
    % For vertex normals, add adjacent face normals, then normalize.  Also
    % add 1/3 of each adjacent area element for vertex area.
    FdA = sqrt(FNdA(:,1).^2 + FNdA(:,2).^2 + FNdA(:,3).^2);
    VdA = zeros(nV, 1);
    for ff = 1:size(Faces, 1) % (This is slow.)
      N(Faces(ff, :), :) = N(Faces(ff, :), :) + FNdA([ff, ff, ff], :);
      VdA(Faces(ff, :), :) = VdA(Faces(ff, :), :) + FdA(ff)/3;
    end
    N = bsxfun(@rdivide, N, sqrt(N(:,1).^2 + N(:,2).^2 + N(:,3).^2));
  end
  
end

% Much faster than using the Matlab version.
function c = CrossProduct(a, b)
  c = [a(:,2).*b(:,3)-a(:,3).*b(:,2), ...
    a(:,3).*b(:,1)-a(:,1).*b(:,3), ...
    a(:,1).*b(:,2)-a(:,2).*b(:,1)];
end








