function s=uplw(s)
if iscell(s)
    s = cellfun(@uplow_str,s,'UniformOutput',0); 
else
    s = uplow_str(s);
end
end
function [s]= uplow_str(s)
s(s=='_' | s=='-') = ' ';
s=lower(s);
idx=regexp([' ' s],'(?<=\s+)\S','start')-1;
s(idx)=upper(s(idx));
end


