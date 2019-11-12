function [  ] = Colormap4(  )
%COLORMAP4 Summary of this function goes here
%   Detailed explanation goes here
    
    DataStruct.Layer.AlphaMap{1} = {[], []};
    DataStruct.Layer.AlphaMap{2} = {[], [], []};
    DataStruct.Layer.AlphaMap{3} = {[]};

    DataStruct.Layer.ColorMap{1} = {[], []};
    DataStruct.Layer.ColorMap{2} = {[], [], []};
    DataStruct.Layer.ColorMap{3} = {[]};

    DataStruct.Layer.ScaledVolAlpha{1} = {[], []};
    DataStruct.Layer.ScaledVolAlpha{2} = {[], [], []};
    DataStruct.Layer.ScaledVolAlpha{3} = {[]};

    DataStruct.Layer.ScaledVolColor{1} = {[], []};
    DataStruct.Layer.ScaledVolColor{2} = {[], [], []};
    DataStruct.Layer.ScaledVolColor{3} = {[]};
    
    DataStruct.Layer.Name{1} = {'LayerA', 'LayerB'};
    DataStruct.Layer.Name{2} = {'LayerC', 'LayerD', 'layerE'};
    DataStruct.Layer.Name{3} = {'LayerF'};
    
    DataStruct.Layer.Info{1} = {[], []};
    DataStruct.Layer.Info{2} = {[], [], []};
    DataStruct.Layer.Info{3} = {[]};
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'
    
    [DataStruct] = AddLayer(DataStruct, [], [2 0]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'
    
    [DataStruct] = MoveLayer(DataStruct, [2 1], [1 0]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'
    
    [DataStruct] = DeleteLayer(DataStruct, [3 3]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'
    
    [DataStruct] = MoveLayer(DataStruct, [4 1], [3 2]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'

    [DataStruct] = MoveLayer(DataStruct, [3 3], [2 2]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'
    
    [DataStruct] = MoveLayer(DataStruct, [2 2], [1 0]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'
    
    [DataStruct] = DeleteLayer(DataStruct, [1 1]);
    
    [displayList, indexList ] = DisplayList(DataStruct);
    
    displayList'

    indexList

end

