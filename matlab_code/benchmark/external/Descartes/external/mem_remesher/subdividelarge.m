function [vnew,fnew] = subdividelarge( vnew, fnew,flag,voriginal,foriginal )

vn=[vnew];
if flag==0
fk1 = fnew(:,1);
fk2 = fnew(:,2);
fk3 = fnew(:,3);

numfaces = (1:size(fnew,1))';

e1=sqrt(sum((vnew(fk1,:)-vnew(fk2,:)).^2,2));
e2=sqrt(sum((vnew(fk1,:)-vnew(fk3,:)).^2,2));
e3=sqrt(sum((vnew(fk2,:)-vnew(fk3,:)).^2,2));

sizzz=mean([e1;e2;e3])+1.96*std([e1;e2;e3]);
else
    
    sizzz=flag;
end
   

while size(vn,1)>1

fk1 = fnew(:,1);
fk2 = fnew(:,2);
fk3 = fnew(:,3);

numfaces = (1:size(fnew,1))';

e1=sqrt(sum((vnew(fk1,:)-vnew(fk2,:)).^2,2));
e2=sqrt(sum((vnew(fk1,:)-vnew(fk3,:)).^2,2));
e3=sqrt(sum((vnew(fk2,:)-vnew(fk3,:)).^2,2));

ed1=sort([fk1 fk2 ]')';
ed2=sort([fk1 fk3 ]')';
ed3=sort([fk2 fk3 ]')';

e1=[e1 numfaces ed1 fk3];
e2=[e2 numfaces ed2 fk2];
e3=[e3 numfaces ed3 fk1];

e=[e1 ; e2 ; e3];

e=e(e(:,1)>sizzz,:);
%single edges

e=sortrows(e,-1);

[etemp1,ia,ic]=unique(e(:,3:4),'rows','stable');
esingle=e(ia,:);

%dubbles
edouble=removerows(e,ia);

[C,ia,ib] = intersect(esingle(:,3:4),edouble(:,3:4),'rows','stable');
% newseries=[esingle(ia,:) edouble(ib,:)];

newseries=[esingle(ia,:) edouble(ib,:)];

newseries=sortrows(newseries,-1);

ind1=(1:2:(2*size(ia,1)-1))';
ind2=(2:2:(2*size(ia,1)))';
ind3=(1:2*size(ia,1))';

test1=ones(2*size(ia,1),1);
test1(ind1)=newseries(:,2);
test1(ind2)=newseries(:,7);

test1(:,2)=ones;
test1(ind1,2)=(1:size(ia))';
test1(ind2,2)=(1:size(ia))';

[etemp1,ia,ic]=unique(test1(:,1),'stable');

test1=(test1(ia,:));

test1(:,3)=ones;
test1(2:end,3)=test1(1:end-1,2);

test1(:,4)=test1(:,3)-test1(:,2);

indicesseries= test1(test1(:,4)==0,2);
indicesseries=unique(indicesseries,'stable');


newseries=newseries(indicesseries,:);


vn=(vnew(newseries(:,3),:)+vnew(newseries(:,4),:)).*0.5;
sizevn=size(vn,1);
indices=size(vnew,1)+(1:sizevn)';

e=[];

e=[horzcat(newseries(:,1:5),indices);horzcat(newseries(:,6:10),indices)];

faces=fnew(e(:,2),:);
n1=vnew(faces(:,1),:)-vnew(faces(:,2),:);
n3=vnew(faces(:,1),:)-vnew(faces(:,3),:);
Normals=cross(n1,n3);
Distance=sqrt(sum((Normals.^2),2));
Normalsoriginal=horzcat(Normals(:,1)./Distance,Normals(:,2)./Distance,Normals(:,3)./Distance);
Normalsoriginal=[Normalsoriginal;Normalsoriginal];

%define new vertices
vnew=[vnew ;vn];
sizevn=size(vn,1);

%define new faces
f1=[e(:,3) e(:,5) e(:,6)];
f2=[e(:,4) e(:,5) e(:,6)];
f=[f1 ; f2];

%correct normals
nn1=vnew(f(:,1),:)-vnew(f(:,2),:);
nn3=vnew(f(:,1),:)-vnew(f(:,3),:);
Normals=cross(nn1,nn3);
Distance=sqrt(sum((Normals.^2),2));
Normals=horzcat(Normals(:,1)./Distance,Normals(:,2)./Distance,Normals(:,3)./Distance);
f(:,4)=sqrt(sum((Normals-Normalsoriginal).^2,2));
f1=f(f(:,4)<1,1:3);
f2=f(f(:,4)>1,1:3);

f2(:,4)=f2(:,2);
f2(:,2)=[];

fnew=[fnew; f1; f2];
fnew=removerows(fnew,e(:,2));


[vnew]=project(vnew, fnew,voriginal,foriginal);
end

% [vnew, fnew]=cleanpatch(vnew, fnew);
% [vnew, fnew,temp] = edgecollaps( vnew, fnew, 0.1 );





