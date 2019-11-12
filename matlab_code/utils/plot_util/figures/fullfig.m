function [ h ] = fullfig(varargin)
%FULLFIG creates a full-screen figure.
%
%% Syntax
%
% fullfig
% fullfig('PropertyName',propertyvalue,...)
% fullfig(h)
% fullfig(...,'Border',BorderPercentage)
% h = fullfig(...)
%
%% Description
%
% fullfig creates a new full-screen graphics figure.  This automatically becomes the
% the current figure and raises it above all other figures on the screen until a
% new figure is either created or called.
%
% fullfig('PropertyName',propertyvalue,...) creates a new figure object using the values
% of the properties specified. For a description of the properties, see Figure Properties.
% MATLAB uses default values for any properties that you do not explicitly define as arguments.
%
% fullfig(h) does one of two things, depending on whether or not a figure with handle h
% exists. If h is the handle to an existing figure, fullfig(h) makes the figure identified
% by h the current figure, makes it visible, makes it full-screen, applies a Border if a Border
% is specified, and raises the figure above all other figures on the screen. The current
% figure is the target for graphics output. If h is not the handle to an existing figure, but
% is an integer, fullfig(h) creates a figure and assigns it the handle h. fullfig(h) where h
% is not the handle to a figure, and is not an integer, is an error.
%
% fullfig(...,'Border',BorderPercentage) creates a Border between the perimeter of the figure
% and the perimeter of your screen. BorderPercentage must be in the range of 0 to 50, and can
% be a scalar value to apply the same percentage value to the width and height of the figure,
% or a two-element vector to apply different Borders in the x- and y- directions, respectively.
%
% h = fullfig(...) returns the handle to the figure object.
%
%% Author Info
% This function was written by Chad A. Greene (www.chadagreene.com) of the
% University of Texas Institute for Geophysics in sunny Austin, Texas,
% October 2014.
%
% See also: figure, Figure Properties, clf, close, axes
% Set defaults:
% SANITY:
h =  findobj('type','figure');
n = length(h);
if n> 20
    error('Cannot open more figures');
end
bufx = 0;
bufy = 0;
NewFigure = true;
SpecifiedNewFigureNumber = false;
% If the first input argument is a handle of a current figure, don't create a new figure:
if nargin>0
    if ishandle(varargin{1})
        NewFigure = false;
        h = varargin{1};
        varargin(1)=[];
    else
        % If the first input argument seems like it might be a figure handle, but isn't a handle
        % of a current figure, create a new figure with the user-specified handle.
        if isnumeric(varargin{1})==1
            SpecifiedNewFigureNumber = true;
            h = varargin{1};
            varargin(1)=[];
        end
    end
end
% Get Border preferences: I originally called this a buffer, but border is more intuitive. Accept either:
tmp = strncmpi(varargin,'bor',3)|strncmpi(varargin,'buf',3);
if any(tmp)
    Border = varargin{find(tmp)+1};
    tmp(find(tmp)+1)=1;
    varargin = varargin(~tmp);
    assert(isnumeric(Border)==1,'Border value must be numeric.')
    assert(numel(Border)<3,'Border must be a scalar or two-element vector.')
    assert(any(Border>=50)==0,'Border value cannot exceed 50 percent.')
    assert(any(Border<0)==0,'Border value cannot be less than zero percent.')
    
    if isscalar(Border)
        bufx = Border/100;
        bufy = Border/100;
    else
        bufx = Border(1)/100;
        bufy = Border(2)/100;
    end
end
% Create new figure or change old one:
if NewFigure
    if SpecifiedNewFigureNumber
        h = set(h,'units','normalized','outerposition',[bufx bufy 1-2*bufx 1-2*bufy],varargin{:});
    else
        h = figure('units','normalized','outerposition',[bufx bufy 1-2*bufx 1-2*bufy],varargin{:});
    end
else
    set(h,'units','normalized','outerposition',[bufx bufy 1-2*bufx 1-2*bufy],varargin{:});
end
set(h,'color','w');
% Clean up:
if nargout==0
    clear h
end
cameratoolbar;
end
