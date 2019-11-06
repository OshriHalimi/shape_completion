function [M, s] = rescale(M)
s = rand()*2;
M.VERT = M.VERT*s;
end
