function [ indexedVol, lastIndex ] = CombineIndex( scaledVolCell, numChannels )
  
    
    indexedVol = zeros(size(scaledVolCell{1})); 
    for c = 1:numChannels
        
        if isa(scaledVolCell{c}, 'uint8')
            indexedVol = indexedVol + ceil( double(scaledVolCell{c})/255*63 ) *64^(c-1);
        else
            indexedVol = indexedVol + ceil( scaledVolCell{c}*63 ) *64^(c-1);
        end
    end
    
    indexedVol(indexedVol==0) = 1;
    
    max(indexedVol(:));
    
    clear scaledVolCell;
    
    lastIndex = 64^numChannels;
    
end

