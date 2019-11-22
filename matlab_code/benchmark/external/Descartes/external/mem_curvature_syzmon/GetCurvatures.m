function [PrincipalCurvatures,PrincipalDir1,PrincipalDir2,FaceCMatrix,VertexCMatrix,Cmagnitude] = GetCurvatures( M,toggleDerivatives )
%% Summary
%Author: Itzik Ben Shabat
%Last Update: July 2014
%Implemented according to "Estimating Curvatures and Their Derivatives on Triangle Meshes" by Szymon Rusinkiewicz (2004)
%and according to its C implementation trimesh2
%{
%GetCurvatures computes the curvature tensor and the principal curvtures at
%each vertex of a mesh given in a face vertex data structure
%INPUT:
%FV -struct - Triangle mesh face vertex data structure (containing FV.face and
%FV.Vertices)
%toggleDerivatives - scalar  1 or 0 indicating wether or not to calcualte curvature derivatives
%OUTPUT:
%PrincipalCurvatures - 2XN matrix (where N is the number of vertices
%containing the proncipal curvtures k1 and k2 at each vertex
%PrincipalDir1 - NX3 matrix containing the direction of the k1 principal
%curvature
%PrincipalDir2 - NX3 matrix containing the direction of the k2 principal
%curvature
%FaceCMatrix - 4XM matrix (where M is the number of faces) containing the 4
%coefficients of the curvature tensr of each face
%VertexCMatrix- 4XN matrix (where M is the number of faces) containing the 4
%coefficients of the curvature tensr of each tensor
%Cmagnitude - NX1 matrix containing the square sum of the curvature tensor coefficients at each
%vertex (invariant scalar indicating the change of curvature)
%}
FaceCMatrix=NaN;
VertexCMatrix=NaN;
Cmagnitude=NaN;

FV = M.fv_struct();

[FaceNormals]=CalcFaceNormals(FV);
[VertexNormals,Avertex,Acorner,up,vp]=CalcVertexNormals(FV,FaceNormals);
%FaceSFM - an mX1 cell matrix (m = number of faces) second fundemental
%VertexSFM - an nX1 cell matrix (n = number of vertices) second fundemental
% wfp - Corner Voronoi Weights 
[FaceSFM,VertexSFM,wfp]=CalcCurvature(FV,VertexNormals,FaceNormals,Avertex,Acorner,up,vp);
[PrincipalCurvatures,PrincipalDir1,PrincipalDir2]=getPrincipalCurvatures(FV,VertexSFM,up,vp);
if toggleDerivatives
    [FaceCMatrix,VertexCMatrix,Cmagnitude]=CalcCurvatureDerivative(FV,FaceNormals,PrincipalCurvatures,up,vp,wfp);
end
end

