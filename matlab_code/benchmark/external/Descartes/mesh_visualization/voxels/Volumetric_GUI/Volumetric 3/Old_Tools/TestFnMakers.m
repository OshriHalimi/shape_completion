function [ ] = Test( )
%TEST Summary of this function goes here
%   Detailed explanation goes here
    

    fnList = {};
    i = 1;
    for i=1:12
        fnList{i} = makeFn(i);
    end
    
    for j=1:12
        disp(fnList{j});
    end
    
    for j=1:12
        disp(fnList{j}());
    end
    
    function [fnHandle] = makeFn(i)
        function [res] = fn()
            res = i+2;
        end
        fnHandle = @fn;
    end
    
end


