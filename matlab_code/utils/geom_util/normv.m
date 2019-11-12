function [ nv ] = normv( v )
nv = sqrt(sum(v.^2,2));
end