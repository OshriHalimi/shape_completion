function [vnew, fnew, temp] = removebadtriangles( vnew, fnew ,voriginal,foriginal)
cases=vnew;
fk1 = fnew(:,1);
fk2 = fnew(:,2);
fk3 = fnew(:,3);

numfaces = (1:size(fnew,1))';

e1=sqrt(sum((vnew(fk1,:)-vnew(fk2,:)).^2,2));
e2=sqrt(sum((vnew(fk1,:)-vnew(fk3,:)).^2,2));
e3=sqrt(sum((vnew(fk2,:)-vnew(fk3,:)).^2,2));

%define area by Heron formula
s=(e1+e2+e3).*0.5;
area=sqrt(s.*(s-e1).*(s-e2).*(s-e3));
quality=[e1 e2 e3];
M = max(quality,[],2);
qualitycheck=area./M;
sizzz=mean(qualitycheck)-1.65*std(qualitycheck);


while size(cases,1)>0

fk1 = fnew(:,1);
fk2 = fnew(:,2);
fk3 = fnew(:,3);

numfaces = (1:size(fnew,1))';

e1=sqrt(sum((vnew(fk1,:)-vnew(fk2,:)).^2,2));
e2=sqrt(sum((vnew(fk1,:)-vnew(fk3,:)).^2,2));
e3=sqrt(sum((vnew(fk2,:)-vnew(fk3,:)).^2,2));

temp(:,1)=mean(e1);
temp(:,2)=std(e1);

%define area by Heron formula
s=(e1+e2+e3).*0.5;
area=sqrt(s.*(s-e1).*(s-e2).*(s-e3));
quality=[e1 e2 e3];
M = max(quality,[],2);
qualitycheck=area./M;
[m,idx]=min(quality,[],2);
score1=find(ismember(idx,[1]));
score2=find(ismember(idx,[2]));
score3=find(ismember(idx,[3]));

ed1=sort([fk1 fk2 ]')';
ed2=sort([fk1 fk3 ]')';
ed3=sort([fk2 fk3 ]')';

e1=[qualitycheck numfaces ed1 ones(size(e1,1),1)];
e2=[qualitycheck numfaces ed2 ones(size(e2,1),1)];
e3=[qualitycheck numfaces ed3 ones(size(e2,1),1)];

 e1(score1,5)=0;
 e2(score2,5)=0;
 e3(score3,5)=0;
 
e=[e1 ; e2 ; e3];
e=e(e(:,5)==0,1:4);
e=e(e(:,1)<sizzz,:);

e=sortrows(e,1);

[etemp,ia,ic]=unique(e(:,3),'rows','stable');
e=e(ia,:);
[etemp,ia,ic]=unique(e(:,4),'rows','stable');
e=e(ia,:);
[test,ia,ic]=unique(e(:,2),'rows','stable');
e=e(ia,:);

ind1=(1:2:(2*size(ia,1)-1))';
ind2=(2:2:(2*size(ia,1)))';
ind3=(1:2*size(ia,1))';

test1=ones(2*size(ia,1),1);
test1(ind1)=e(:,3);
test1(ind2)=e(:,4);

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


e=e(indicesseries,:);

cases=e(:,3:4);
 
averages=(vnew(cases(:,1),:)+vnew(cases(:,2),:)).*0.5;
vnew(cases(:,1),:)=averages;
vnew(cases(:,2),:)=averages;
    
[vnew, fnew]=cleanpatch(vnew, fnew);
[vnew]=project(vnew, fnew,voriginal,foriginal);
end