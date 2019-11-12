function [ DataStruct ] = DeleteLayer( DataStruct, index )
%DELETELAYER Deletes a layer in the DataStruct;

% finished product:
% DataStruct.Master.Alpha
% DataStruct.Master.Color
% DataStruct.Master.volIndexAlpha
% DataStruct.Master.volIndexColor
%
% working data:
% DataStruct.Layer.AlphaMap
% DataStruct.Layer.ColorMap
% DataStruct.Layer.ScaledVolAlpha
% DataStruct.Layer.ScaledVolColor
% DataStruct.Layer.Info. name, path, 



    
    
    % this could be a problem area
    numG = length(DataStruct.Layer.Name{index(1)});
    
    if numG==1
        DataStruct.Layer.AlphaMap = Delete(DataStruct.Layer.AlphaMap, index(1));
        DataStruct.Layer.ColorMap = Delete(DataStruct.Layer.ColorMap, index(1));
        DataStruct.Layer.ScaledVolAlpha = Delete( DataStruct.Layer.ScaledVolAlpha, index(1));
        DataStruct.Layer.ScaledVolColor = Delete(DataStruct.Layer.ScaledVolColor, index(1));
        DataStruct.Layer.Name = Delete(DataStruct.Layer.Name, index(1));
        DataStruct.Layer.Info = Delete(DataStruct.Layer.Info, index(1));
    else numG>1
        DataStruct.Layer.AlphaMap{index(1)} = Delete(DataStruct.Layer.AlphaMap{index(1)}, index(2));
        DataStruct.Layer.ColorMap{index(1)} = Delete(DataStruct.Layer.ColorMap{index(1)}, index(2));
        DataStruct.Layer.ScaledVolAlpha{index(1)} = Delete( DataStruct.Layer.ScaledVolAlpha{index(1)}, index(2));
        DataStruct.Layer.ScaledVolColor{index(1)} = Delete(DataStruct.Layer.ScaledVolColor{index(1)}, index(2));
        DataStruct.Layer.Name{index(1)} = Delete(DataStruct.Layer.Name{index(1)}, index(2));
        DataStruct.Layer.Info{index(1)} = Delete(DataStruct.Layer.Info{index(1)}, index(2));
    end
    

end

function [cellList] = Delete(cellList, index)

    cellList = [cellList(1:(index-1)) cellList((index+1):end)];
    
end

