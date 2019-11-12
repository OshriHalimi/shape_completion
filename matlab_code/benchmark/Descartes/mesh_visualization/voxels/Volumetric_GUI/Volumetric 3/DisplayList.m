function [ displayList, indexList ] = DisplayList( DataStruct )
% Creates a string cell array that can be displayed in a ListBox, such that
% it communicates the structure of DataStruct.

    infos = DataStruct.Layer.Name;

    numGroups = length(DataStruct.Layer.Name);

    displayList = cell(1,0);
    indexList = [];
    
    for g = 1:numGroups
        
        group = infos{g};
        
        numLay = length(group);
        
        if numLay>1
            displayList = [displayList {'Blend'} ];
            indexList = [indexList; [g 0]];
            for l = 1:numLay
                displayList = [displayList {['   -' infos{g}{l}]} ];
                indexList = [indexList; [g l]];
            end
        else
            displayList = [displayList {[infos{g}{1}]} ];
            indexList = [indexList; [g 1]];
        end
            
    end
    
end

