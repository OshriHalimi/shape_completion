function [] = video_append()
global vid
writeVideo(vid,getframe(gcf));
end