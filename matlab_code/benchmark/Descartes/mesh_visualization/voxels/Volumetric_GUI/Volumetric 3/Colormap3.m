function [ ] = Colormap3( )
%COLORMAP3 Summary of this function goes here
%   Detailed explanation goes here

    [X, Y, Z] = meshgrid(1:100, 1:100, 1:100);
    
    Xsmall = X(1:50, 1:100, 1:100);
    
    save X.mat X;
    save Y.mat Y;
    save Z.mat Z;
    save Xsmall.mat Xsmall;
    
    
    Xscaled = X/max(X(:));
    Yscaled = Y/max(Y(:));
    Zscaled = Z/max(Z(:));

    volScaledC{1}{1} = Xscaled.*(Yscaled>.5);
    volScaledC{1}{2} = Yscaled.*(Yscaled>.5);
    volScaledC{2}{1} = Zscaled;

    volScaledA{1}{1} = Zscaled.*(Yscaled>.5);
    volScaledA{1}{2} = Zscaled.*(Yscaled>.5);
    volScaledA{2}{1} = 1-Zscaled;
    
    mapsCell{1}{1} = [(0:(1/63):1)' 1-(0:(1/63):1)' 1-(0:(1/63):1)' ];
    mapsCell{1}{2} = [1-(0:(1/63):1)' (0:(1/63):1)' 1-(0:(1/63):1)' ];
    mapsCell{2}{1} = [1-(0:(1/63):1)' 1-(0:(1/63):1)'  (0:(1/63):1)'];
    
    aMapsCell{1}{1} = (0:(1/63):1)' * .5;
    aMapsCell{1}{2} = (0:(1/63):1)' * .5;
    aMapsCell{2}{1} = (0:(1/63):1)' * .5;
    
    
    masterVolIndC = MakeIndex(volScaledC);
    masterVolIndA = MakeIndex(volScaledA);
    
    
    
    %speed test
    masterCmap = MakeMap(mapsCell);
    masterAmap = MakeMap(aMapsCell);
    
    
    CA(:,:,:,1) = masterVolIndC;
    CA(:,:,:,2) = masterVolIndA;
    
    
    fig1 = figure();
    viewAxis = axes();


    [slices1, slices2, slices3] = volumeRenderMono(CA, [],[],[], viewAxis);

    daspect([1 1 1]);

    set(slices1, 'visible', 'on');
    
    %min(masterVolInd(:))
    %max(masterVolInd(:))
    size(masterCmap)
    size(masterAmap)

    set(fig1,'colormap',masterCmap, 'alphamap', masterAmap);
    set(viewAxis, 'projection', 'perspective');

    
    for i = 1:50
        aMapsCell{2}{1} = (0:(1/63):1)' * .5*(1-i/50);
        masterAmap = MakeMap(aMapsCell);
        set(fig1,'alphamap', masterAmap);
        pause(.01);
    end
    
    for i = 1:50

        mapsCell{1}{1} = [(0:(1/63):1)' 1-(0:(1/63):1)' 1-(0:(1/63):1)' ] * (1-i/50) + [1-(0:(1/63):1)' (0:(1/63):1)' 1-(0:(1/63):1)' ] * (i/50);
        masterCmap = MakeMap(mapsCell);
        set(fig1,'colormap', masterCmap);
        pause(.01);
    end
    
end

