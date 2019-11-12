function [ masterVolIndex ] = MakeIndex( inputVolCell )
%MAKEINDEX 
    
%    {}     treat as single overiding colormap
%    {}{}{} treat as blend colormaps 
% nested cell array    

    numGroups = length(inputVolCell);
    
    volS = size(inputVolCell{1}{1});
    
    masterVolIndex = uint32(ones(size(inputVolCell{1}{1})));

    lastInd = 0; % index 1 is always transparent
    
    for g = 1:numGroups
        
        g
        
        numChan = length(inputVolCell{g});
        
        [tempVolInd, newLastInd] = CombineIndex(inputVolCell{g}, numChan);
        
        trans = masterVolIndex==1 & tempVolInd~=1;
        
        masterVolIndex(trans) = tempVolInd(trans) + lastInd;
        
        clear tempVolInd;
        
        lastInd = newLastInd+lastInd;
        
    end
    
    clear lastInd;
    
    masterVolIndex = uint32(masterVolIndex);

    
    min(masterVolIndex(:))
    max(masterVolIndex(:))
    
end

