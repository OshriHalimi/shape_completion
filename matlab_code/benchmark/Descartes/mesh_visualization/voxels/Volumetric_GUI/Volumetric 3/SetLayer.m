function [ DataStruct ] = SetLayer( DataStruct, layerStruct, index )

    DataStruct.Layer.AlphaMap{index(1)}{index(2)} = layerStruct.AlphaMap;
    DataStruct.Layer.ColorMap{index(1)}{index(2)} = layerStruct.ColorMap;
    DataStruct.Layer.ScaledVolAlpha{index(1)}{index(2)} = layerStruct.ScaledVolAlpha;
    DataStruct.Layer.ScaledVolColor{index(1)}{index(2)} = layerStruct.ScaledVolColor;
    DataStruct.Layer.Name{index(1)}{index(2)} = layerStruct.Name;
    DataStruct.Layer.Info{index(1)}{index(2)} = layerStruct.Info;

    clear layerStruct;
end

