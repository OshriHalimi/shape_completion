function [FaceSFM,VertexSFM,wfp]=CalcCurvature(FV,VertexNormals,FaceNormals,Avertex,Acorner,up,vp)
%% Summary
%Author: Itzik Ben Shabat
%Last Update: July 2014

%CalcFaceCurvature recives a list of vertices and faces in FV structure
%and the normal at each vertex and calculates the second fundemental
%matrix and the curvature using least squares
%INPUT:
%FV - face-vertex data structure containing a list of vertices and a list of faces
%VertexNormals - nX3 matrix (n=number of vertices) containing the normal at each vertex
%FaceNormals - mX3 matrix (m = number of faces) containing the normal of each face
%OUTPUT:
%FaceSFM - an mX1 cell matrix (m = number of faces) second fundemental
%VertexSFM - an nX1 cell matrix (n = number of vertices) second fundemental
%wfp - corner voronoi weights
%% Code
% disp('Calculating Curvature Tensors... Please wait');
%matrix of each face at each cell
FaceSFM=cell(size(FV.faces,1),1);
VertexSFM=cell(size(FV.vertices,1),1);
[FaceSFM{1:end,1}]=deal(zeros(2,2));
[VertexSFM{1:end,1}]=deal(zeros(2,2));
% Get all edge vectors
e0=FV.vertices(FV.faces(:,3),:)-FV.vertices(FV.faces(:,2),:);
e1=FV.vertices(FV.faces(:,1),:)-FV.vertices(FV.faces(:,3),:);
e2=FV.vertices(FV.faces(:,2),:)-FV.vertices(FV.faces(:,1),:);
% Normalize edge vectors
e0_norm=normr(e0);
e1_norm=normr(e1);
e2_norm=normr(e2);

wfp=zeros(size(FV.faces,1),3);

for i=1:size(FV.faces,1)
    %Calculate Curvature Per Face
    %set face coordinate frame
    nf=FaceNormals(i,:);
    t=e0_norm(i,:)';
    B=cross(nf,t)';
    B= B/norm(B);
    %extract relevant normals in face vertices
    n0=VertexNormals(FV.faces(i,1),:);
    n1=VertexNormals(FV.faces(i,2),:);
    n2=VertexNormals(FV.faces(i,3),:);
    %solve least squares problem of th form Ax=b
    A=[e0(i,:)*t e0(i,:)*B 0;
        0 e0(i,:)*t e0(i,:)*B;
        e1(i,:)*t e1(i,:)*B 0;
        0 e1(i,:)*t e1(i,:)*B;
        e2(i,:)*t e2(i,:)*B 0;
        0 e2(i,:)*t e2(i,:)*B];
    b=[(n2-n1)*t;(n2-n1)*B;(n0-n2)*t;(n0-n2)*B;(n1-n0)*t;(n1-n0)*B];
    %[LA,DA] = ldl(A'*A);
    % bA=A'*b;
    %  x = LA'\(DA\(LA\bA));
    x=A\b;
    
    FaceSFM{i,1}=[x(1),x(2);x(2) x(3)];
    Kn(i)=[1 0]*FaceSFM{i,1}*[1;0];
    
    %Calculate Curvature Per Vertex
    %calculate voronoi weights
    wfp(i,1)=Acorner(i,1)/Avertex(FV.faces(i,1));
    wfp(i,2)=Acorner(i,2)/Avertex(FV.faces(i,2));
    wfp(i,3)=Acorner(i,3)/Avertex(FV.faces(i,3));
    %Calculate new coordinate system and project the tensor
    for j=1:3
        [new_ku,new_kuv,new_kv]=ProjectCurvatureTensor(t,B,nf,x(1),x(2),x(3),up(FV.faces(i,j),:),vp(FV.faces(i,j),:));
        VertexSFM{FV.faces(i,j),1}= VertexSFM{FV.faces(i,j),1}+wfp(i,j)*[new_ku new_kuv;new_kuv new_kv];
    end
end
% disp('Finished Calculating Curvature Tensors.');
end
