function [ DataStruct ] = AddLayer( DataStruct, addition, index )
%ADDLAYER adds a layer to the DataStruct;

% finished product:
% DataStruct.Master.Alpha
% DataStruct.Master.Color
% DataStruct.Master.VolIndexAlpha
% DataStruct.Master.VolIndexColor
%
% working data:
% DataStruct.Layer.AlphaMap
% DataStruct.Layer.ColorMap
% DataStruct.Layer.ScaledVolAlpha
% DataStruct.Layer.ScaledVolColor
% DataStruct.Layer.Info. name, path, 

    


    if ~isempty(DataStruct)&&(~isempty(DataStruct.Layer.Name));
        
        alphaMap = (0:1/63:1)';
        colorMap = [1-(0:(1/63):1)' (0:(1/63):1)' 1-(0:(1/63):1)' ];
        newAlphaMap{1} = alphaMap;
        newColorMap{1} = colorMap;
        newScaledVolAlpha{1} = [];
        newScaledVolColor{1} = [];
        newName{1} = 'New Layer';
        newInfo{1}.PathAlphaVol = 'Not Set'; 
        newInfo{1}.PathColorVol = 'Not Set'; 
        newInfo{1}.Filter = '';
        newInfo{1}.Bounds = [0 1];

        if ~isempty(addition)
            newAlphaMap{1} = addition.AlphaMap;
            newColorMap{1} = addition.ColorMap;
            newScaledVolAlpha{1} = addition.ScaledVolAlpha;
            newScaledVolColor{1} = addition.ScaledVolColor;
            newName{1} = addition.Name;
            newInfo{1} = addition.Info;
        end

        numG = length(DataStruct.Layer.Name{index(1)});

        if index(2) == 0 || numG>=3
            DataStruct.Layer.AlphaMap = Add(DataStruct.Layer.AlphaMap, newAlphaMap, index(1));
            DataStruct.Layer.ColorMap = Add(DataStruct.Layer.ColorMap, newColorMap, index(1));
            DataStruct.Layer.ScaledVolAlpha = Add( DataStruct.Layer.ScaledVolAlpha, newScaledVolAlpha, index(1));
            DataStruct.Layer.ScaledVolColor = Add(DataStruct.Layer.ScaledVolColor, newScaledVolColor, index(1));
            DataStruct.Layer.Name = Add(DataStruct.Layer.Name, newName, index(1));
            DataStruct.Layer.Info = Add(DataStruct.Layer.Info, newInfo, index(1));
        elseif index(2)>0
            DataStruct.Layer.AlphaMap{index(1)} = Add(DataStruct.Layer.AlphaMap{index(1)}, newAlphaMap{1}, index(2));
            DataStruct.Layer.ColorMap{index(1)} = Add(DataStruct.Layer.ColorMap{index(1)}, newColorMap{1}, index(2));
            DataStruct.Layer.ScaledVolAlpha{index(1)} = Add( DataStruct.Layer.ScaledVolAlpha{index(1)}, newScaledVolAlpha{1}, index(2));
            DataStruct.Layer.ScaledVolColor{index(1)} = Add(DataStruct.Layer.ScaledVolColor{index(1)}, newScaledVolColor{1}, index(2));
            DataStruct.Layer.Name{index(1)} = Add(DataStruct.Layer.Name{index(1)}, newName{1}, index(2));
            DataStruct.Layer.Info{index(1)} = Add(DataStruct.Layer.Info{index(1)}, newInfo{1}, index(2));
        end
    
    else % INITIALIZE NEW DATA STRUCT WITH A BASE LAYER
        % DEFAULTS
        alphaMap = (0:1/63:1)';
        colorMap = [(0:(1/63):1)' 1-(0:(1/63):1)' 1-(0:(1/63):1)' ];
        DataStruct.Layer.AlphaMap{1} = {alphaMap};
        DataStruct.Layer.ColorMap{1} = {colorMap};
        DataStruct.Layer.ScaledVolAlpha{1} = {[]};
        DataStruct.Layer.ScaledVolColor{1} = {[]};
        DataStruct.Layer.Name{1} = {'Base Layer'};
        DataStruct.Layer.Info{1}{1}.PathAlphaVol = 'Not Set';
        DataStruct.Layer.Info{1}{1}.PathColorVol = 'Not Set';
        DataStruct.Layer.Info{1}{1}.Filter = [];
        DataStruct.Layer.Info{1}{1}.Bounds = [0 1];
    end

end

function [cellList] = Add(cellList, addition, index)

    cellList = [cellList(1:(index-1)) {addition} cellList(index:end)];
    
end

