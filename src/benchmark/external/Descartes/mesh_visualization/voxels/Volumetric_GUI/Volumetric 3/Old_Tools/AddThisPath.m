%Author: James W. Ryland
%June 14, 2012

function [ ] = AddThisPath( )
%ADDTHISPATH this function adds this directory to the MATLAB path list.
%   I started writing the method summeries in reverse order so this is the 
%   last one. YAY!!! Now to decide if I want to take the plunge to go
%   through and comment lines for each file.
    
    w = what();
    addpath(w.path);

end
