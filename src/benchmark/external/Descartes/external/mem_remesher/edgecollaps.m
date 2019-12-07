function [vnew, fnew,temp] = edgecollaps( vnew, fnew, sizzz,voriginal,foriginal )

[vnew, fnew]=cleanpatch(vnew, fnew);
cases=vnew;
sizzz=sizzz/sqrt(3);
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

ed1=sort([fk1 fk2 ]')';
ed2=sort([fk1 fk3 ]')';
ed3=sort([fk2 fk3 ]')';

e1=[e1 numfaces ed1 ];
e2=[e2 numfaces ed2 ];
e3=[e3 numfaces ed3 ];

e=[e1 ; e2 ; e3];
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

% doubles= find(ismember(e(:,4),e(:,3)));
% e=removerows(e,doubles);

cases=e(:,3:4);
 
averages=(vnew(cases(:,1),:)+vnew(cases(:,2),:)).*0.5;
vnew(cases(:,1),:)=averages;
vnew(cases(:,2),:)=averages;
    
[vnew, fnew]=cleanpatch(vnew, fnew);

[vnew]=project(vnew, fnew,voriginal,foriginal);
end

