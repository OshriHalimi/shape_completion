function smoothsurf( x,y,z,xn,yn,varargin )
%smoothsurf Plot a smooth surface using kernel smoothing. Input 3d points (x,y,z)
%  optional parameters: 'xn','yn','method','size','interpolant'
%  xn/yn: generating surface using xn by yn gridlines (default is 20 by 20)
%  method: the convolution smoothing kernel, can be either: 'gaussian' or 'box' (default)
%  size: size of the convolution kernel, default is 3
%  interpolant: 'nearest'(default),'linear' or 'natural'
%
%  Version 1.1
%  Author: Liutong Zhou
%  3/1/2017

params.method ='box';
params.size = 3;
params.interpolant='nearest';
% overwrite defaults
params = parse_pv_pairs(params,varargin);
% check the parameters for acceptability
params = check_params(params);
if exist('xn','var')~=1||isempty(xn)
    xn=20;
end
if exist('yn','var')~=1||isempty(yn)
    yn=20;
end
x=row2column(x);
y=row2column(y);
z=row2column(z);
F=scatteredInterpolant(x,y,z,params.interpolant);
x=linspace(min(x),max(x),xn);
y=linspace(min(y),max(y),yn);
[X,Y]=meshgrid(x,y);
Z=F(X,Y);
data = smooth3(cat(3,Z,Z),params.method,params.size);
surf(X,Y,data(:,:,1))
axis tight
colormap jet;shading interp;
%camlight headlight
lighting gouraud
xlabel('x','FontSize',13);ylabel('y','FontSize',13);
title('Title','FontSize',14);
c=colorbar;c.Label.String = 'My Colorbar Label';c.Label.FontSize = 12;
end
%% update defaults
function params=parse_pv_pairs(params,pv_pairs)
% parse_pv_pairs: parses sets of property value pairs, allows defaults
% usage: params=parse_pv_pairs(default_params,pv_pairs)
%
% arguments: (input)
%  default_params - structure, with one field for every potential
%             property/value pair. Each field will contain the default
%             value for that property. If no default is supplied for a
%             given property, then that field must be empty.
%
%  pv_array - cell array of property/value pairs.
%             Case is ignored when comparing properties to the list
%             of field names. Also, any unambiguous shortening of a
%             field/property name is allowed.
%
% arguments: (output)
%  params   - parameter struct that reflects any updated property/value
%             pairs in the pv_array.
%
% Example usage:
% First, set default values for the parameters. Assume we
% have four parameters that we wish to use optionally in
% the function examplefun.
%
%  - 'viscosity', which will have a default value of 1
%  - 'volume', which will default to 1
%  - 'pie' - which will have default value 3.141592653589793
%  - 'description' - a text field, left empty by default
%
% The first argument to examplefun is one which will always be
% supplied.
%
%   function examplefun(dummyarg1,varargin)
%   params.Viscosity = 1;
%   params.Volume = 1;
%   params.Pie = 3.141592653589793
%
%   params.Description = '';
%   params=parse_pv_pairs(params,varargin);
%   params
%
% Use examplefun, overriding the defaults for 'pie', 'viscosity'
% and 'description'. The 'volume' parameter is left at its default.
%
%   examplefun(rand(10),'vis',10,'pie',3,'Description','Hello world')
%
% params =
%     Viscosity: 10
%        Volume: 1
%           Pie: 3
%   Description: 'Hello world'
%
% Note that capitalization was ignored, and the property 'viscosity'
% was truncated as supplied. Also note that the order the pairs were
% supplied was arbitrary.
npv = length(pv_pairs);
n = npv/2;
if n~=floor(n)
    error 'Property/value pairs must come in PAIRS.'
end
if n<=0
    % just return the defaults
    return
end
if ~isstruct(params)
    error 'No structure for defaults was supplied'
end
% there was at least one pv pair. process any supplied
propnames = fieldnames(params);
lpropnames = lower(propnames);
for i=1:n
    p_i = lower(pv_pairs{2*i-1});
    v_i = pv_pairs{2*i};
    ind = find(strcmp(p_i,lpropnames));
    if isempty(ind)
        ind = find(strncmp(p_i,lpropnames,length(p_i)));
        if isempty(ind)
            error(['No matching property found for: ',pv_pairs{2*i-1}])
        elseif length(ind)>1
            error(['Ambiguous property name: ',pv_pairs{2*i-1}])
        end
    end
    p_i = propnames{ind};
    % override the corresponding default in params
    params = setfield(params,p_i,v_i); %#ok
end
end
%% validate parameters
function params = check_params(params)
% check the parameters for acceptability
% check method
valid = {'box','gaussian'};
ind = find(strncmpi(params.method,valid,length(params.method)));
if (length(ind)==1)
    params.method = valid{ind};
else
    error(['Invalid values for method: ',params.method])
end
% check size
if isempty(params.size)
    params.size = 3;
else
    if ~isscalar(params.size)||params.size<1
        error 'size must be positive integer'
    end
end
% check interpolant
valid = {'nearest','linear','natural'};
ind = find(strncmpi(params.interpolant,valid,length(params.interpolant)));
if (length(ind)==1)
    params.interpolant = valid{ind};
else
    error(['Invalid values for interpolant: ',params.interpolant])
end
end
%% row to column
function x=row2column(x)
if isrow(x)
    x=x';
else
    return
end
end