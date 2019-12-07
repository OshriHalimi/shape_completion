%% This script is an example of how to implement the functions for estimating curvature and their derivatives 
%Author: Itzik Ben Shabat
%Last Update: July 2014

%%
%Clear all variables, close all windows and clear command window 
clear all
close all
clc
%% Generate the example triangle mesh example
StepSize=0.5;
GridSize=20;
grid=1:StepSize:GridSize;
[x,y]=meshgrid(grid,grid);
FV.faces = delaunay(x,y);
z = peaks(size(grid,2));
FV.vertices=[x(1:end)',y(1:end)',z(1:end)'];

%% calcualte curvatures
getderivatives=0;
[PrincipalCurvatures,PrincipalDir1,PrincipalDir2,FaceCMatrix,VertexCMatrix,Cmagnitude]= GetCurvatures( FV ,getderivatives);

GausianCurvature=PrincipalCurvatures(1,:).*PrincipalCurvatures(2,:);
%% Draw the mesh to the screen 
figure('name','Triangle Mesh Curvature Example','numbertitle','off','color','w');
colormap cool
caxis([min(GausianCurvature) max(GausianCurvature)]); % color overlay the gaussian curvature
mesh_h=patch(FV,'FaceVertexCdata',GausianCurvature','facecolor','interp','edgecolor','interp','EdgeAlpha',0.2);
%set some visualization properties
set(mesh_h,'ambientstrength',0.35);
axis off
view([-45,35.2]);
camlight();
lighting phong
colorbar();


