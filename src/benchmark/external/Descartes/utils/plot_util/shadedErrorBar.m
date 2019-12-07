function varargout=shadedErrorBar(x,y,errBar,varargin)
% generate continuous error bar area around a line plot
%
% function H=shadedErrorBar(x,y,errBar, ...)
%
% Purpose 
% Makes a 2-d line plot with a pretty shaded error bar made
% using patch. Error bar color is chosen automatically.
%
%
% Inputs (required)
% x - vector of x values [optional, can be left empty]
% y - vector of y values or a matrix of n observations by m cases
%     where m has length(x);
% errBar - if a vector we draw symmetric errorbars. If it has a size
%          of [2,length(x)] then we draw asymmetric error bars with
%          row 1 being the upper bar and row 2 being the lower bar
%          (with respect to y -- see demo). ** alternatively ** 
%          errBar can be a cellArray of two function handles. The 
%          first defines statistic the line should be and the second 
%          defines the error bar.
%
% Inputs (optional, param/value pairs)
% 'lineProps' - ['-k' by default] defines the properties of
%             the data line. e.g.:    
%             'or-', or {'-or','markerfacecolor',[1,0.2,0.2]}
% 'transparent' - [true  by default] if true, the shaded error
%               bar is made transparent. However, for a transparent
%               vector image you will need to save as PDF, not EPS,
%               and set the figure renderer to "painters". An EPS 
%               will only be transparent if you set the renderer 
%               to OpenGL, however this makes a raster image.
% 'patchSaturation'- [0.2 by default] The saturation of the patch color.
%
%
%
% Outputs
% H - a structure of handles to the generated plot objects.
%
%
% Examples:
% y=randn(30,80); 
% x=1:size(y,2);
%
% 1)
% shadedErrorBar(x,mean(y,1),std(y),'lineprops','g');
%
% 2)
% shadedErrorBar(x,y,{@median,@std},'lineprops',{'r-o','markerfacecolor','r'});
%
% 3)
% shadedErrorBar([],y,{@median,@(x) std(x)*1.96},'lineprops',{'r-o','markerfacecolor','k'});
%
% 4)
% Overlay two transparent lines:
% clf
% y=randn(30,80)*10; 
% x=(1:size(y,2))-40;
% shadedErrorBar(x,y,{@mean,@std},'lineprops','-r','transparent',1);
% hold on
% y=ones(30,1)*x; y=y+0.06*y.^2+randn(size(y))*10;
% shadedErrorBar(x,y,{@mean,@std},'lineprops','-b','transparent',1);
% hold off
%
%
% Rob Campbell - November 2009



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse input arguments
narginchk(3,inf)

params = inputParser;
params.CaseSensitive = false;
params.addParameter('lineProps', '-k', @(x) ischar(x) | iscell(x));
params.addParameter('transparent', true, @(x) islogical(x) || x==0 || x==1);
params.addParameter('patchSaturation', 0.2, @(x) isnumeric(x) && x>=0 && x<=1);

params.parse(varargin{:});

%Extract values from the inputParser
lineProps =  params.Results.lineProps;
transparent =  params.Results.transparent;
patchSaturation = params.Results.patchSaturation;

if ~iscell(lineProps), lineProps={lineProps}; end


%Process y using function handles if needed to make the error bar dynamically
if iscell(errBar) 
    fun1=errBar{1};
    fun2=errBar{2};
    errBar=fun2(y);
    y=fun1(y);
else
    y=y(:).';
end

if isempty(x)
    x=1:length(y);
else
    x=x(:).';
end


%Make upper and lower error bars if only one was specified
if length(errBar)==length(errBar(:))
    errBar=repmat(errBar(:)',2,1);
else
    s=size(errBar);
    f=find(s==2);
    if isempty(f), error('errBar has the wrong size'), end
    if f==2, errBar=errBar'; end
end

if length(x) ~= length(errBar)
    error('length(x) must equal length(errBar)')
end


%Log the hold status so we don't change
initialHoldStatus=ishold;
if ~initialHoldStatus, hold on,  end

H = makePlot(x,y,errBar,lineProps,transparent,patchSaturation);

if ~initialHoldStatus, hold off, end

if nargout==1
    varargout{1}=H;
end



function H = makePlot(x,y,errBar,lineProps,transparent,patchSaturation)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot to get the parameters of the line

    H.mainLine=plot(x,y,lineProps{:});


    % Work out the color of the shaded region and associated lines.
    % Here we have the option of choosing alpha or a de-saturated
    % solid colour for the patch surface.
    mainLineColor=get(H.mainLine,'color');
    edgeColor=mainLineColor+(1-mainLineColor)*0.55;

    if transparent
        faceAlpha=patchSaturation;
        patchColor=mainLineColor;
    else
        faceAlpha=1;
        patchColor=mainLineColor+(1-mainLineColor)*(1-patchSaturation);
    end


    %Calculate the error bars
    uE=y+errBar(1,:);
    lE=y-errBar(2,:);


    %Add the patch error bar



    %Make the patch
    yP=[lE,fliplr(uE)];
    xP=[x,fliplr(x)];

    %remove nans otherwise patch won't work
    xP(isnan(yP))=[];
    yP(isnan(yP))=[];


    if(isdatetime(x))
        H.patch=patch(datenum(xP),yP,1,'HandleVisibility','off');
    else
        H.patch=patch(xP,yP,1,'HandleVisibility','off');
    end

    set(H.patch,'facecolor',patchColor, ...
        'edgecolor','none', ...
        'facealpha',faceAlpha)


    %Make pretty edges around the patch. 
    H.edge(1)=plot(x,lE,'-','color',edgeColor,'HandleVisibility','off');
    H.edge(2)=plot(x,uE,'-','color',edgeColor,'HandleVisibility','off');



    uistack(H.mainLine,'top') % Bring the main line to the top
end

function H=demo_shadedErrorBar
% demo function for shadedErrorBar
%
% function demo_shadedErrorBar
%
% No input or output arguments.
%
%
% 


rows=2;
cols=2;

clf



subplot(rows,cols,1)
%Plot the mean and standard deviation then overlay the raw data
y=randn(30,80)*5;
x=(1:size(y,2))-40;
yP = sin( linspace(-2*pi,2*pi,length(x)) )*20;
y = bsxfun(@plus,y,yP)+60;
shadedErrorBar(x,y,{@mean,@std}); 

hold on
plot(x,y,'.','color',[0.5,0.5,0.95])

hold off

grid on




subplot(rows,cols,2)
%Overlay different lines (transparent) lines and change their properties
hold on

y=ones(30,1)*x; 
y=y+0.06*y.^2+randn(size(y))*10;
shadedErrorBar(x,y,{@mean,@std},'lineprops','-b','patchSaturation',0.33)
shadedErrorBar(x,2*y+20,{@mean,@std},'lineprops',{'-go','MarkerFaceColor','g'});

y=randn(30,80)*5; 
x=(1:size(y,2))-40;
yP = sin( linspace(-2*pi,2*pi,length(x)) )*20;
y = bsxfun(@plus,y,yP)+60;

%Make red line non-transparent
shadedErrorBar(x, y, {@mean,@std}, 'lineprops', '-r','transparent',false,'patchSaturation',0.075)
hold off

grid on




subplot(rows,cols,3)
% Post-hoc modifications of line properties


y=randn(30,80)*5; 
x=(1:size(y,2));
yP = sin( linspace(-2*pi,2*pi,length(x)) )*20;
y = bsxfun(@plus,y,yP);


s = shadedErrorBar(x, y, {@mean,@std}, 'lineprops', '-r');

%Set face and edge properties
set(s.edge,'LineWidth',2,'LineStyle',':')
s.mainLine.LineWidth = 5;
s.patch.FaceColor = [0.5,0.25,0.25];

hold on
plot(s.mainLine.XData, s.mainLine.YData,'or','MarkerFaceColor','w')

hold off
grid on

set(gca,'XTickLabel',[],'YTickLabel',[])





subplot(rows,cols,4)
% Post-hoc modifications of line properties


y=randn(256,80)*5; 
x=(1:size(y,2));
yP = cos( linspace(-2*pi,2*pi,length(x)) )*10;
y = bsxfun(@plus,y,yP);


shadedErrorBar(x, y, {@mean,@std}, 'lineprops', '-r')

hold on

y=mean(y)+16;
errbar = [2*ones(1,length(x)) ; 4*ones(1,length(x))];

shadedErrorBar(x, y, errbar, 'lineprops', '-g')

end

