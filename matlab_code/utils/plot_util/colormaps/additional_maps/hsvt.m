function cm_data=hsvt(m)

cm = hsv(2048); 
cm = circshift(cm,round(2048/20));
if nargin < 1
    cm_data = cm;
else
    clr=rgb2hsv(cm);
    clr(153:end,1)=clr(153:end,1)+1; % hardcoded
    cm_data=interp1(linspace(0,1,size(cm,1)),clr,linspace(0,1,m));
    cm_data(cm_data(:,1)>1,1)=cm_data(cm_data(:,1)>1,1)-1;
    cm_data=hsv2rgb(cm_data);
end