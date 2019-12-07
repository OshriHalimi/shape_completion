function [] = video_init(oname,fr)

global vid 
% Make unique 
appender = 1; 
name = oname; 
while isfile(['./' name '.avi'])
    name = sprintf('%s_(%d)',oname,appender);
    appender = appender+1;
end

if ~exist('name','var'); name = 'my_animation'; end
if ~exist('fr','var'); fr = 30; end

vid = VideoWriter(name);
vid.FrameRate = fr; %30 is default
vid.Quality = 100; % 75 is default
open(vid); 

end