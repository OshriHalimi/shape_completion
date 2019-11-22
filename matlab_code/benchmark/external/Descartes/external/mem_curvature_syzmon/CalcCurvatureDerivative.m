function [FaceCMatrix,VertexCMatrix,Cmagnitude]=CalcCurvatureDerivative(FV,FaceNormals,PrincipalCurvatures,up,vp,wfp)
%Author: Itzik Ben Shabat
%Last Update: July 2014
%% Summary
%{ 
%CalcCurvatureDerivative recives a list of vertices and faces in FV structure
%and the curvature tensor at each vertex and calculates thecurvature
%derivative matrix (2X2X2) VertexCMatrix using least squares
%INPUT:
%FV - face-vertex data structure containing a list of vertices and a list of faces
%VertexSFM - nX(2X2) cell array (n=number of vertices) containing the second funcdmental tensor at each vertex
%FaceSFM -  mX(2X2) cell array (m=number of faces) containing the second fundmental tensor at each face
%OUTPUT:
%FaceCMatrix - an mX(2X2X2) cell matrix (m = number of faces) second
%fundemental derivative tensor at each face
%VertexCMatrix -  an nX(2X2X2) cell matrix (n = number of vertices) second
%fundemental derivative tensor at each vertex
%}
%% Code
% disp('Calculating C Tensors... Please wait');
%variable initialization
FaceCMatrix=zeros(size(FV.faces,1),4);
VertexCMatrix=zeros(size(FV.vertices,1),4);
new_CMatrix=zeros(1,4);
FV_SFM=cell(3,1);
[FV_SFM{1:end,1}]=deal(zeros(2,2,2));

Cmagnitude=zeros(size(FV.vertices,1),1);

% Get all edge vectors
e0=FV.vertices(FV.faces(:,3),:)-FV.vertices(FV.faces(:,2),:);
e1=FV.vertices(FV.faces(:,1),:)-FV.vertices(FV.faces(:,3),:);
e2=FV.vertices(FV.faces(:,2),:)-FV.vertices(FV.faces(:,1),:);
% Normalize edge vectors
e0_norm=normr(e0);
e1_norm=normr(e1);
e2_norm=normr(e2);

for i=1:size(FV.faces,1)
    %Calculate C Per Face
    %set face coordinate frame
    nf=FaceNormals(i,:);
    t=e0_norm(i,:)';
    B=cross(nf,t)';
    B= B/norm(B);
    
    %solve least squares problem of th form Ax=b
    u(1,1)=e0(i,:)*t;
    u(1,2)=e1(i,:)*t;
    u(1,3)=e2(i,:)*t;
    v(1,1)=e0(i,:)*B;
    v(1,2)=e1(i,:)*B;
    v(1,3)=e2(i,:)*B;
    
    
    for j=1:3
        np=cross(up(FV.faces(i,j),:),vp(FV.faces(i,j),:));
        np=np/norm(np);
        k1=PrincipalCurvatures(1,FV.faces(i,j));
        k2=PrincipalCurvatures(2,FV.faces(i,j));
        [new_ku,new_kuv,new_kv]=ProjectCurvatureTensor(up(FV.faces(i,j),:)',vp(FV.faces(i,j),:)',np,k1,0,k2,t',B');
        %vertex second fundemental matrix in the face coordinate frame
        FV_SFM{j,1}=[new_ku new_kuv;new_kuv new_kv];
    end
    Delta_e=[FV_SFM{3,1}(1,1)-FV_SFM{2,1}(1,1) FV_SFM{1,1}(1,1)-FV_SFM{3,1}(1,1) FV_SFM{2,1}(1,1)-FV_SFM{1,1}(1,1) ];
    Delta_f=[FV_SFM{3,1}(1,2)-FV_SFM{2,1}(1,2) FV_SFM{1,1}(1,2)-FV_SFM{3,1}(1,2) FV_SFM{2,1}(1,2)-FV_SFM{1,1}(1,2) ];
    Delta_g=[FV_SFM{3,1}(2,2)-FV_SFM{2,1}(2,2) FV_SFM{1,1}(2,2)-FV_SFM{3,1}(2,2) FV_SFM{2,1}(2,2)-FV_SFM{1,1}(2,2) ];
    
    sumU2=u*u';
    sumV2=v*v';
    sumUV=u*v';
    A=[sumU2        sumUV             0             0;
        sumUV 2*sumU2+sumV2 2*sumUV 0;
        0 2*sumUV sumU2+2*sumV2 sumUV;
        0          0           sumUV             sumV2];
    b=[Delta_e*u'; Delta_e*v'+2*Delta_f*u'; 2*Delta_f*v'+Delta_g*u';Delta_g*v'];
    x=A\b;

    FaceCMatrix(i,:)=x;
    %Calculate new coordinate system and project the tensor
    for j=1:3
        new_CMatrix=ProjectCTensor(t,B,nf,x,up(FV.faces(i,j),:),vp(FV.faces(i,j),:));
        VertexCMatrix(FV.faces(i,j),:)= VertexCMatrix(FV.faces(i,j),:)+wfp(i,j)*new_CMatrix;
    end
end
    a=VertexCMatrix(:,1);
    b=VertexCMatrix(:,2);
    c=VertexCMatrix(:,3);
    d=VertexCMatrix(:,4);
    Cmagnitude(:,1)=a.^2+3.*b.^2+3.*c.^2+d.^2;
    epsilon=10^-5;
    Cmagnitude(Cmagnitude<epsilon)=0;

% disp('Finished Calculating C Tensors.');
end
