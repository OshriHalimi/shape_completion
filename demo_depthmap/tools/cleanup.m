function [M, is_outlier] = cleanup(S)
    
missing = setdiff(1:size(S.VERT,1), unique(S.TRIV(:)));
is_outlier = false(size(S.VERT,1),1);
is_outlier(missing) = true;
M = removeVertices(S, is_outlier, false);

fprintf('%d outliers detected.\n', sum(is_outlier));

end
