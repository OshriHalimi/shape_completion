function [FaceNormals]=CalcFaceNormals(FV)
%% Summary
%Author: Itzik Ben Shabat
%Last Update: July 2014

%CalcFaceNormals recives a list of vrtexes and faces in FV structure
% and calculates the normal at each face and returns it as FaceNormals
%INPUT:
%FV - face-vertex data structure containing a list of vertices and a list of faces
%OUTPUT:
%FaceNormals - an nX3 matrix (n = number of faces) containng the norml at each face
%% Code
% Get all edge vectors
e0=FV.vertices(FV.faces(:,3),:)-FV.vertices(FV.faces(:,2),:);
e1=FV.vertices(FV.faces(:,1),:)-FV.vertices(FV.faces(:,3),:);
% Calculate normal of face
FaceNormals=cross(e0,e1);
FaceNormals=normr(FaceNormals);
end

