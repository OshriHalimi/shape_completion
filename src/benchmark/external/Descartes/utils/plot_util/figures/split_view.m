function split_view(n,remove_titles)
if ~exist('n','var'); n = 2; end
if ~exist('remove_titles','var'); remove_titles = 0; end
% TODO: Insert support for figures that already have subplots in them 
nr = floor(sqrt(n));
nc = ceil(n/nr);
warning('off'); 
for i=1:n
    if i==1
        s = subplot_tight(nr,nc,i,[0.01,0.01],gca);
        if remove_titles
            title(''); 
        end
    else
        h = copyobj(s, gcf ,'legacy');
        s = subplot_tight(nr,nc, i,[0.01,0.01],h);
    end
end
warning('on');
end