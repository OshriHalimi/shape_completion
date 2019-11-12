function print_list(l,header)
if exist('header','var'); cprintf('*Keywords','%s\n',header); 
for i=1:length(l)
    fprintf('%d.\t',i); cprintf('*Text',sprintf('%s\n',l{i})); 
end
end