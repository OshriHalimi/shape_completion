function [C] = TestArrayFun()
%TESTARRAYFUN Summary of this function goes here
%   Detailed explanation goes here
    
    A = [1 2; 3 4];

    A = mat2cell(A);
    
    function [b] = afn( a)
        b = {a/2, a/2};
    end
    
    C = cellfun(@afn,A,'UniformOutput',0);
    
    %cell2mat(C);
end
