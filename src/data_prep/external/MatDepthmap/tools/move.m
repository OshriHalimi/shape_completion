function [M, Rx, Ry, Rz, T] = move(M)
Rx = rotate_x(rand()*360);
Ry = rotate_y(rand()*360);
Rz = rotate_z(rand()*360);
M.VERT = M.VERT*Rx;
M.VERT = M.VERT*Ry;
M.VERT = M.VERT*Rz;
d = max(range(M.VERT));
T = [rand()*d, rand()*d, rand()*d];
M.VERT = M.VERT + repmat(T, M.n, 1);
end
