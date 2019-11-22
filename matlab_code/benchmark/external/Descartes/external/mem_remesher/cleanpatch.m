function [vnew, fnew]=cleanpatch(V, F)

%remove duplicate vertices

 [vnew, indexm, indexn] =  unique(V, 'rows');
fnew = indexn(F);

%remove nonsens faces
numfaces = (1:size(fnew,1))';

e1=fnew(:,1)-fnew(:,2);
e2=fnew(:,1)-fnew(:,3);
e3=fnew(:,2)-fnew(:,3);

e1=[e1 numfaces];
e2=[e2 numfaces];
e3=[e3 numfaces];

e1=e1(e1(:,1)==0,2);
e2=e2(e2(:,1)==0,2);
e3=e3(e3(:,1)==0,2);

nonsensefaces=unique(vertcat(e1,e2,e3));

fnew=removerows(fnew,nonsensefaces);

% remove nonconnected vertices
numvertices = (1:size(vnew,1))';
connected=unique(reshape(fnew,3*size(fnew,1),1));
numvertices=removerows(numvertices,connected);
vtemp=vnew;
vtemp=removerows(vtemp,numvertices);
[lia,loc]=ismember(vnew,vtemp,'rows');

fnew = loc(fnew);
vnew=vtemp;

% 




