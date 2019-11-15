function [  ] = AlphaScaleChecker()
%ALPHASCALECHECKER Summary of this function goes here
%   Detailed explanation goes here

    figure();

    l = 15;
    
    x = 0.0:.01:1;
    
    for j=1:length(x)
        a = x(j);
        y1(j) = a;
        for i = 1:(l-1)
            y1(j) = y1(j) + (1-y1(j))*a; 
        end
    end
    
    y2 = 1-(1-x).^l; 
    
    subplot(1,4,1);
    plot(x, y1)
    
    subplot(1,4,2);
    plot(x, y2, 'red')
    
    
    subplot(1,4,3);
    plot(y1, x, 'blue')
    
    x1 = -(-y1+1).^(1/l)+1;
    
    subplot(1,4,4);
    plot(y1, x1, 'red')
    
    
    diff = (y1-y2);
    disp(diff);
    
end

