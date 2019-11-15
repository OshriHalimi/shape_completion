function [ C ] = mdaTimes( A, B, collapseDim, gpu)
% this function performs tensor-like multiplication, where a specific
% dimension is indacated for collapsing over. The order of the dimesnions
% matters here, and the collapse dim will become singleton. This function
% does not reorder the dimensions of the output matrix.

% there are several places in this program that could be significantly
% improved in terms of readability and understanding by using mdaTimes
% instead of a custom on the spot magic trick featuring all sorts of hard
% to debug mental gymnastics.

% additionally this function should be fast on GPU and CPU
    
    if isempty(gpu)
        gpu = 'no';
    end
    
    aIndNum = length(size(A));
    bIndNum = length(size(B));
    maxInd = max([length(size(A)) length(size(B))]);
    aSize = ones(1, maxInd);
    bSize = ones(1, maxInd);
    aSize(1, 1:aIndNum) = size(A);
    bSize(1, 1:bIndNum) = size(B);
    
    if strcmp(gpu, 'yes')
        A = gpuArray(A);
        B = gpuArray(B);
    else
        maxI = max([aSize; bSize],[], 1);
        
        Arep = -ones(1, maxInd);
        Arep(aSize==1) = maxI(aSize==1);
        Arep(aSize==maxI) = 1;
        Brep = -ones(1, maxInd);
        Brep(bSize==1) = maxI(bSize==1);
        Brep(bSize==maxI) = 1;
        
        if (min(Arep)==(-1))||(min(Brep)==(-1))
            disp('Dimesnsion Mismatch, will throw non-integer exception');
            disp('Dimesnions must match, or be singleton');
        end
        
        A = repmat(A, Arep);
        B = repmat(B, Brep);
        
    end
    
    C = [];
    
    if strcmp(gpu, 'yes')
        C = arrayfun(@multiply, A, B);
        C = gather(C);
    else
        C = A.*B;
    end
    
    if ~isempty(collapseDim)&&(collapseDim>=1)
        C = sum(C, collapseDim);
    end
    
    function [c] = multiply(a, b)
        c = a*b;
    end

end

