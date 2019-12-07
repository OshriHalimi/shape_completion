function [ layerStruct ] = GetLayer( DataStruct, index )
    
    layerStruct.AlphaMap = DataStruct.Layer.AlphaMap{index(1)}{index(2)};
    layerStruct.ColorMap = DataStruct.Layer.ColorMap{index(1)}{index(2)};
    layerStruct.ScaledVolAlpha = DataStruct.Layer.ScaledVolAlpha{index(1)}{index(2)};
    layerStruct.ScaledVolColor = DataStruct.Layer.ScaledVolColor{index(1)}{index(2)};
    layerStruct.Name = DataStruct.Layer.Name{index(1)}{index(2)};
    layerStruct.Info = DataStruct.Layer.Info{index(1)}{index(2)};
    
end

