function [ map ] = CombineMap( mapsCell, numMaps)
%COMBINEMAP should work with colormaps and alpha maps

    
    %colMaps need to be 1 by 64 long
    if isempty(mapsCell)
        mapsCell{1} = [(0:(1/63):1)' 1-(0:(1/63):1)' 1-(0:(1/63):1)' ];
        mapsCell{2} = [1-(0:(1/63):1)' (0:(1/63):1)' 1-(0:(1/63):1)' ];
        mapsCell{3} = [1-(0:(1/63):1)' 1-(0:(1/63):1)'  (0:(1/63):1)'];
    end
    
    
    map = [];

    if numMaps == 1

        map = mapsCell{1};
        
    elseif numMaps == 2

        map1 = repmat(mapsCell{1}, [64 1]);
        map2 = repmat(mapsCell{2}, [1 1 64]);
        map2 = permute(map2, [3 1 2]);
        map2 = reshape(map2, 64*64, size(map1, 2));
        
        map = (map1+map2)/2; % simple average
                            % should make weighted by luminance
                            
        clear map1;
        clear map2;
        

    elseif numMaps == 3

        map1 = repmat(mapsCell{1}, [64^2 1]);
        map2 = repmat(mapsCell{2}, [1 1 64]);
        map2 = permute(map2, [3 1 2]);
        map2 = reshape(map2, 64^2, size(map1, 2));
        map2 = repmat(map2, [64 1]);
        map3 = repmat(mapsCell{3}, [1 1 64^2]);
        map3 = permute(map3, [3 1 2]);
        map3 = reshape(map3, 64^3, size(map1, 2));
        
        map = (map1+map2+map3)/3;
        clear map1;
        clear map2;
        clear map3;
    end
    
    clear mapsCell;
     

end

