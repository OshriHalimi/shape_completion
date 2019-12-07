clearvars; close all; clc;
%%
%change this to the 3d Matrix you want to display:
load('spherev.mat'); 
input= sphere;
alpha = 1 ; % transparency
%------------------------
disp('calculating...');

%cropping the matrix ( reduces lag for large empty matrices)
[yy,xx]=find(sum(input,3));
xmin = min(xx(:));
xmax = max(xx(:));
ymin = min(yy(:));
ymax = max(yy(:));
[xx,zz] = find(squeeze(sum(input,1)));
zmin=min(zz(:));
zmax = max(zz(:));
testcrop=input(ymin:ymax,xmin:xmax,zmin:zmax);


%uncomment this to shrink the volume of the clusters
testcrop=meltVolume(testcrop,18,0.7);

%color different clusters differently:
% testcrop=colorVoxelGroups(double(testcrop));

% prepareFig(2,'3D');
% figuresize( 30 , 30,'cm'  );
% subplot = @(m,n,p) subtightplot(m,n,p,[0.04 0.04], [0.04 0.04], [0.04 0.04]);
fullfig;
voxelSurf(testcrop,true,[1 size(testcrop,1) 1 size(testcrop,2) 1 size(testcrop,3)],alpha );

%use this if you want to have a separate color for each voxelgroup:
% colormap(lines(max(testcrop(:))));

% colormap(lines(64));

disp('done');