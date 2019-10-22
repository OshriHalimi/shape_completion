function [N, depth, matches] = create_rangemap(M, w, h, max_edge)

d1 = sqrt(sum((M.VERT(M.TRIV(:,1),:)-M.VERT(M.TRIV(:,2),:)).^2,2));
d2 = sqrt(sum((M.VERT(M.TRIV(:,1),:)-M.VERT(M.TRIV(:,3),:)).^2,2));
d3 = sqrt(sum((M.VERT(M.TRIV(:,3),:)-M.VERT(M.TRIV(:,2),:)).^2,2));
res = median([d1;d2;d3]);

params.ccd_width = 30;        % CCD width in world coordinates
params.resolution = [w h];    % rangemap resolution [cols rows]

[depth, matches] = snapshot(M.VERT', M.TRIV', params.resolution, 'ortho', -1, params.ccd_width);

% figure, plot_mesh(M)
% figure, imagesc(rangemap), axis equal, colormap(gray), colorbar

assert(~any(any(depth==0)))

ccd_height = params.ccd_width * h / w;
min_x = -params.ccd_width/2;
max_x = params.ccd_width/2;
min_y = -ccd_height/2;
max_y = ccd_height/2;
step_x = (max_x - min_x) / (w-1);
step_y = (max_y - min_y) / (h-1);

%rangemap(isnan(rangemap))=0; 
x = step_x*repmat(1:h, 1, w)';
y = step_y*kron(1:w, ones(1,h))';
z = depth(:);
matches = matches(:);

t1 = (1:((w-1)*h))';
t1(h:h:end,:) = [];
t2 = t1+h;
t3 = t1+1;

N.VERT = [x y -z];
N.TRIV = [t1 t2 t3 ; t2 t2+1 t3];

% M.VERT = M.VERT * [1 0 0 ; 0 -1 0 ; 0 0 1];

keep = ~isnan(N.VERT(:,3));

N = removeVertices(N,~keep,0);
%figure, plot_mesh(M)
matches = matches(keep);

t = max_edge*res;

d1 = sqrt(sum((N.VERT(N.TRIV(:,1),:)-N.VERT(N.TRIV(:,2),:)).^2,2));
d2 = sqrt(sum((N.VERT(N.TRIV(:,1),:)-N.VERT(N.TRIV(:,3),:)).^2,2));
d3 = sqrt(sum((N.VERT(N.TRIV(:,3),:)-N.VERT(N.TRIV(:,2),:)).^2,2));

keep = d1 < t & d2 < t & d3 < t;

N.TRIV = N.TRIV(keep,:);
N.m = size(N.TRIV,1);

% fprintf('Keeping largest connected component... ');

% S = N;
% % compute adjacency matrix
% A = sparse(...
%     [S.TRIV(:,1); S.TRIV(:,2); S.TRIV(:,3)], ...
%     [S.TRIV(:,2); S.TRIV(:,3); S.TRIV(:,1)], ...
%     ones(3 * S.m, 1), ...
%     S.n, S.n, 3 * S.m);
% 
% % keep largest connected component
% [nComponents,sizes,members] = networkComponents(A);
% if nComponents>1
%     v = true(1,S.n);
%     [~, ci] = max(sizes);
%     v(members{ci})=0;
%     S = removeVertices(S,v,0);
%     matches = matches(~v);
% end
% N = S;

% fprintf('done.\n');

end
