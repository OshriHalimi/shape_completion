function [  ] = MakeXYZ( dims )
%MAKEXYZ Summary of this function goes here
%   Detailed explanation goes here

    [X, Y, Z] = meshgrid(1:dims(2),1:dims(1),1:dims(3));

    indVols = {'X', 'Y', 'Z'};
    
    fileTypes = {'.mat'};
    
    [filename, pathname] = uiputfile(fileTypes);
        
    inds = {'X_' 'Y_' 'Z_'}
    
    if ~(isnumeric(filename) && filename==0)
        for i = 1:3
            save([pathname inds{i} filename], indVols{i});
        end
    end 

end

