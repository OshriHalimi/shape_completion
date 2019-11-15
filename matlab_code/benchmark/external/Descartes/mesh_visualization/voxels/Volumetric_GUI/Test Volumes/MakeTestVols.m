function [ ] = MakeTestVols( )
%MAKETESTVOLS Summary of this function goes here
%   Detailed explanation goes here

    [X, Y, Z] = meshgrid(1:100, 1:100, 1:100);
    
    Xsmall = X(1:50, 1:100, 1:100);
    
    save X.mat X;
    save Y.mat Y;
    save Z.mat Z;
    save Xsmall.mat Xsmall;
    
    sphere = zeros(100,100,100);
    
    center = 50;
    sphere = ( (X-center).^2 + (Y-center).^2 + (Z-center).^2).^.5 <(center-5);
    
    sphere = smooth3(sphere,'gaussian', [5 5 5]);
    
    save sphere.mat sphere;
    
    shell = sphere.*(sphere>.1).*(sphere<.9);
    
    shell = smooth3(shell,'gaussian', [5 5 5]);
    
    save shell.mat shell;
    
    slab = X.*(X<50).*(X>25);
    
    slabCut = slab.*shell;

    save slab.mat slab;
    save slabCut.mat slab;
    
    
    [mriX, mriY, mriZ] = meshgrid(1:256, 1:160, 1:256);
    
    save mriX.mat mriX;
    save mriY.mat mriY;
    save mriZ.mat mriZ;
    
end

