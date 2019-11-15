function [ passCheck, error ] = CheckDataStruct( DataStruct )

%    {}     treat as single overiding colormap
%    {}{}{} treat as blend colormaps 
% nested cell array    

% right now only reports the first error

    numGroups = length(DataStruct.Layer.AlphaMap);
    
    passCheck = 1;
    error = {};
    i = 1;
    
    for g = 1:numGroups
        
        numLayers = length(DataStruct.Layer.AlphaMap{g});
        
        for l = 1:numLayers
            
            
            
            if (~isempty(DataStruct.Layer.ScaledVolColor{g}{l}))&&(~isempty(DataStruct.Layer.ScaledVolAlpha{g}{l}))
            
                colSize = size(DataStruct.Layer.ScaledVolColor{g}{l});
                alpSize = size(DataStruct.Layer.ScaledVolAlpha{g}{l});
                mainSize = size(DataStruct.Layer.ScaledVolAlpha{1}{1});

                if ~(length(colSize)==length(alpSize) && sum(colSize==alpSize)==3)

                    passCheck = 0;
                    error{i} = ['Dimension Mis-match between Alpha Source and Color Source: ' DataStruct.Layer.Name{g}{l} ' [' mat2str(colSize) '][' mat2str(alpSize) ']' ];
                    i = i+1;
                    
                end
                
                if ~(length(colSize)==length(mainSize) && sum(colSize==mainSize)==3)
                    
                    passCheck = 0;
                    error{i} = ['Dimension Mis-match between 1st layer and ' DataStruct.Layer.Name{g}{l}  ' [' mat2str(mainSize) '][' mat2str(colSize) ']'];
                    i = i+1;
                    
                end
                
            else
                
                passCheck = 0;
                error{i} = ['Source Not Set: ' DataStruct.Layer.Name{g}{l}];
                i = i+1;
                
            end

        end
        
        
    end

    for i=1:length(error);
        disp(error{i});
    end
    
    %passCheck
    
end

