function s = trunc_struct(s,len,inds)
fns = fieldnames(s);
for fn = fns.'
    a = s.(fn{1}); 
    if isstruct(a) 
        s.(fn{1}) = trunc_struct(a,len,inds); 
    else
        if length(size(a))==2 && ( ((size(a,1)==1  && size(a,2)==len)) || ((size(a,2)==1  && size(a,1)==len)) )
            s.(fn{1}) = a(inds); 
        end
    end
end
end
