function [ Mout ] = vmul( M, v )
% Hadamard multiplication of M by the vector v. 

if size(v,2) == size(M,2)
    Mout = M .* repmat(v, size(M,1), 1);
else
    Mout = M .* repmat(v, 1, size(M,2));
end


end
