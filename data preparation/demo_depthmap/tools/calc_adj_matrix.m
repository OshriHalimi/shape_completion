function A = calc_adj_matrix(M, sym)

n = size(M.VERT,1);
m = size(M.TRIV,1);

% asymmetric
A = sparse(...
    [M.TRIV(:,1); M.TRIV(:,2); M.TRIV(:,3)], ...
    [M.TRIV(:,2); M.TRIV(:,3); M.TRIV(:,1)], ...
    1, ...
    n, n, 3 * m);

if nargin==2 && sym
    A = A+A';
    A = double(A~=0);
end

end
