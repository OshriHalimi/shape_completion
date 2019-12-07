function [ DataStruct ] = MoveLayer( DataStruct, index, toInd )
%Move layer moves a layer in the DataStruct;

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
% DataStruct.Layer.Name
% DataStruct.Layer.Info  stores structs


    index
    toInd

    addition.ColorMap = [];
    addition.ScaledVolAlpha = [];
    addition.ScaledVolColor = [];
    addition.Name = [];
    addition.Info = [];
    
    if index(2)~=0
        addition.AlphaMap = DataStruct.Layer.AlphaMap{index(1)}{index(2)};
        addition.ColorMap = DataStruct.Layer.ColorMap{index(1)}{index(2)};
        addition.ScaledVolAlpha = DataStruct.Layer.ScaledVolAlpha{index(1)}{index(2)};
        addition.ScaledVolColor = DataStruct.Layer.ScaledVolColor{index(1)}{index(2)};
        addition.Name = DataStruct.Layer.Name{index(1)}{index(2)};
        addition.Info = DataStruct.Layer.Info{index(1)}{index(2)};
    end
    
    numG = length(DataStruct.Layer.Name{index(1)});
    numGto =length(DataStruct.Layer.Name{toInd(1)});
    
    % if I use the second toInd index wisely I will not have to make a blend with function
    % simply do not check numGto and rely on toInd;
    
    if (numG == 1 || index(2)==0) && ( toInd(2)==0)
        DataStruct.Layer.AlphaMap = Move(DataStruct.Layer.AlphaMap, index(1), toInd(1));
        DataStruct.Layer.ColorMap = Move(DataStruct.Layer.ColorMap, index(1), toInd(1));
        DataStruct.Layer.ScaledVolAlpha = Move( DataStruct.Layer.ScaledVolAlpha, index(1), toInd(1));
        DataStruct.Layer.ScaledVolColor = Move(DataStruct.Layer.ScaledVolColor, index(1), toInd(1));
        DataStruct.Layer.Name = Move(DataStruct.Layer.Name, index(1), toInd(1));
        DataStruct.Layer.Info = Move(DataStruct.Layer.Info, index(1), toInd(1));
    
    elseif (numG > 1 || index(2)>0) && ( toInd(2)>0 )
        DataStruct = AddLayer(DataStruct, addition, toInd);
        DataStruct = DeleteLayer(DataStruct, index);
    
    elseif (numG == 1 && index(2)==0) && ( toInd(2)>0)
        DataStruct = AddLayer(DataStruct, addition, toInd);
        DataStruct = DeleteLayer(DataStruct, index);
    
    elseif (numG > 1 || index(2)>0) && ( toInd(2)==0)
        DataStruct = DeleteLayer(DataStruct, index);
        DataStruct = AddLayer(DataStruct, addition, toInd);
        
    end
end

function [cellList] = Move(cellList, index, toIndex)

    
        order = [];
        for i = 1:length(cellList);
            if i==toIndex
                order = [order index];
            end
            if i~=index
                order = [order i];
            end
        end
        cellList = cellList(order);
        
end

