% ParforProgressbar   Progress monitor for `parfor` loops
%    ppm = ParforProgressbar(numIterations) constructs a ParforProgressbar object.
%    'numIterations' is an integer with the total number of
%    iterations in the parfor loop.
%
%    ppm = ParforProgressbar(___, 'showWorkerProgress', true) will display
%    the progress of all workers (default: false).
%    
%    ppm = ParforProgressbar(___, 'progressBarUpdatePeriod', 1.5) will
%    update the progressbar every 1.5 second (default: 1.0 seconds).
%
%    ppm = ParforProgressbar(___, 'title', 'my fancy title') will
%    show 'my fancy title' on the progressbar.
%
%    ppm = ParforProgressbar(___, 'parpool', 'local') will
%    start the parallel pool (parpool) using the 'local' profile.
%
%    ppm = ParforProgressbar(___, 'parpool', {profilename, poolsize, Name, Value}) 
%    will start the parallel pool (parpool) using the profilename profile with
%    poolsize workers and any Name Value pair supported by function parpool.
%
%
%    <strong>Usage:</strong>
%    % 'numIterations' is an integer with the total number of iterations in the loop.
%    numIterations = 100000;
%
%    % Then construct a ParforProgMon object:
%    ppm = ParforProgressbar(numIterations);
%
%    parfor i = 1:numIterations
%       % do some parallel computation
%       pause(100/numIterations);
%       % increment counter to track progress
%       ppm.increment();
%    end
%
%   % Delete the progress handle when the parfor loop is done.
%   delete(ppm);
%
%
% Based on <a href="https://de.mathworks.com/matlabcentral/fileexchange/60135-parfor-progress-monitor-progress-bar-v3">ParforProgMonv3</a>.
% Uses the progressbar from: <a href="https://de.mathworks.com/matlabcentral/fileexchange/6922-progressbar">progressbar</a>.
classdef ParforProgressbar < handle
   % These properties are the same for the server and worker and are not
   % subject to change 
   properties ( GetAccess = private, SetAccess = private )
      ServerPort % Port of the server connection
      ServerName % host name of the server connection
      totalIterations % Total number of iterations (entire parfor)
      numWorkersPossible % Total number of possible workers
      stepSize uint64 % Number of steps before update
   end
   
   % These properties are completely different between server and each
   % worker.
   properties (Transient, GetAccess = private, SetAccess = private)
       it uint64 % worker: iteration
       UserData % Anything the user wants to store temporarily in the worker

       workerTable % server: total progress with ip and port of each worker 
       showWorkerProgress logical% server: show not only total progress but also the estimated progress of each worker
       timer % server: timer object
       progressTotalOld % server: Current total progress (float between 0 and 1).

       isWorker logical % server/worker: This identifies a worker/server
       connection % server/worker: udp connection
   end

   properties (Transient, GetAccess = public, SetAccess = private)
       workerID uint64 % worker: unique id for each worker
   end

   methods ( Static )
      function o = loadobj( X )
%          import libUtil.ParforProgressbar;
         % loadobj - METHOD Reconstruct a ParforProgressbar object
         
         % Once we've been loaded, we need to reconstruct ourselves correctly as a
         % worker-side object.
         debug('LoadObj');
         o = ParforProgressbar( {X.ServerName, X.ServerPort, X.totalIterations, X.numWorkersPossible, X.stepSize, X.UserData} );
      end
   end
   
   methods
       function o = ParforProgressbar( numIterations, varargin )
%            import libUtil.progressbar;
           % ParforProgressbar - CONSTRUCTOR Create a ParforProgressbar object
           % 
           %    ppb = ParforProgressbar(numIterations)
           %    numIterations is an integer with the total number of
           %    iterations in the parfor loop.
           %
           %    ppm = ParforProgressbar(___, 'showWorkerProgress', true) will display
           %    the progress of all workers (default: false).
           %    
           %    ppm = ParforProgressbar(___, 'progressBarUpdatePeriod', 1.5) will
           %    update the progressbar every 1.5 second (default: 1.0 seconds).
           %
           %    ppm = ParforProgressbar(___, 'title', 'my fancy title') will
           %    show 'my fancy title' on the progressbar).
           %
           %    ppm = ParforProgressbar(___, 'parpool', 'local') will
           %    start the parallel pool (parpool) using the 'local' profile.
           %
           %    ppm = ParforProgressbar(___, 'parpool', {profilename, poolsize, Name, Value}) 
           %    will start the parallel pool (parpool) using the profilename profile with
           %    poolsize workers and any Name Value pair supported by function parpool.
           %
           if iscell(numIterations) % worker
               debug('Start worker.');
               host = numIterations{1};
               port = numIterations{2};
               o.totalIterations = numIterations{3};
               o.numWorkersPossible = numIterations{4};
               o.stepSize = numIterations{5};
               o.UserData = numIterations{6};
               o.ServerName = host;
               o.ServerPort = port;
               t = getCurrentTask(); 
               o.workerID = t.ID;
               % Connect the worker to the server, so that we can send the
               % progress to the server.
               o.connection = udp(o.ServerName, o.ServerPort); 
               fopen(o.connection);
               o.isWorker = true; 
               o.it = 0; % This is the number of iterations this worker is called.
               debug('Send login cmd');
               % Send a login request to the server, so that the ip and
               % port can be saved by the server. This is neccessary to
               % close each worker when the parfor loop is finished.
               fwrite(o.connection,[o.workerID, 0],'ulong'); % login to server
           else % server
               % - Server constructor
               p = inputParser;
               
               showWorkerProgressDefault = false;
               progressBarUpdatePeriodDefault = 1.0;
               titleDefault = '';
               poolDefault = '';
               
               validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
               is_valid_profile = @(x) ischar(x) || iscell(x);
               addRequired(p,'numIterations', validScalarPosNum );
               addParameter(p,'showWorkerProgress', showWorkerProgressDefault, @isscalar);
               addParameter(p,'progressBarUpdatePeriod', progressBarUpdatePeriodDefault, validScalarPosNum);
               addParameter(p,'title',titleDefault,@ischar);
               addParameter(p,'parpool',poolDefault,is_valid_profile)
               parse(p,numIterations, varargin{:});               
               o.showWorkerProgress = p.Results.showWorkerProgress;
               o.totalIterations = p.Results.numIterations;
               o.progressTotalOld = 0;
               ppool = p.Results.parpool;
               
               debug('Start server.');
               pPool = gcp('nocreate');   
               if isempty(pPool)
                   if isempty(ppool)
                       pPool = parpool; % Create new parallel pool with standard setting
                   elseif ischar(ppool)
                       pPool = parpool(ppool); % Create parallel pool with given profilename
                   elseif iscell(ppool)
                       pPool = parpool(ppool{:}); % Create parallel pool with given input arguments.
                   end
               else
                   % A parallel pool is still running. Let's keep it.
               end
               o.numWorkersPossible = pPool.NumWorkers;
               
               % We don't send each progress step to the server because
               % this will slow down each worker. Insead, we send the
               % progress each stepSize iterations.
               if (o.totalIterations / o.numWorkersPossible) > 200
                   % We only need to resolve 1% gain in worker progress
                   % progressStepSize = worker workload/100
                   progressStepSize = floor(o.totalIterations/o.numWorkersPossible/100);
               else
                   % We will transmit the progress each step.
                   progressStepSize = 1;
               end
               o.stepSize = progressStepSize;
               pct = pctconfig;
               o.ServerName = pct.hostname;
               % Create server connection to receive the updates from each
               % worker via udp. receiver is called each time a data
               % package is received with this class object handle to keep
               % track of the progress.
               o.connection = udp(o.ServerName, 'DatagramReceivedFcn', {@receiver, o}, 'DatagramTerminateMode', 'on', 'EnablePortSharing', 'on');
               fopen(o.connection);
               
               % This new connection uses a free port, which we have to
               % provide to each worker to connect to.
               o.ServerPort = o.connection.LocalPort;
               o.workerTable = table('Size',[pPool.NumWorkers, 4],'VariableTypes',{'uint64','string','uint32','logical'},'VariableNames',{'progress','ip','port','connected'});
               o.isWorker = false;
               
               % Open a progressbar with 0% progress and optionally
               % initiallize also the progress of each worker with 0%.
               % Also optionally, provide a title to the main progress
               if o.showWorkerProgress
                   titles = cell(pPool.NumWorkers + 1, 1);
                   if ~any(contains(p.UsingDefaults,'title'))
                       titles{1} = p.Results.title;
                   else
                       titles{1} = 'Total progress';
                   end
                   for i = 1 : pPool.NumWorkers
                       titles{i+1} = sprintf('Worker %d', i);
                   end
                   progressbar(titles{:});
               else
                   if ~any(contains(p.UsingDefaults,'title'))
                       progressbar(p.Results.title);
                   else
                       progressbar;
                   end
               end
               % Start a timer and update the progressbar periodically
               o.timer = timer('BusyMode','drop','ExecutionMode','fixedSpacing','StartDelay',p.Results.progressBarUpdatePeriod*2,'Period',p.Results.progressBarUpdatePeriod,'TimerFcn',{@draw_progress_bar, o});
               start(o.timer);
               o.UserData = {};
           end
       end
       
       function o = saveobj( X )
           debug('SaveObj');
           o.ServerPort = X.ServerPort;
           o.ServerName = X.ServerName;
           o.totalIterations = X.totalIterations;
           o.numWorkersPossible = X.numWorkersPossible;
           o.stepSize = X.stepSize;
           o.UserData = X.UserData;
       end
       
       function delete( o )
           debug('Delete object');
           o.close();
       end
       
       function increment( o )
           o.it = o.it + 1;
           if mod(o.it, o.stepSize) == 0
               debug('Send it=%d',o.it);
               fwrite(o.connection,[o.workerID, o.it], 'ulong');
           end
       end
       
       function UserData = getUserData( o )
           UserData = o.UserData;
       end

       function setUserData( o, UserData )
           o.UserData = UserData;
       end

       function close( o )
%            import libUtil.progressbar;
           % Close worker/server connection
           if isa(o.connection, 'udp')
               if strcmp(o.connection.Status, 'open')
                    debug('Close worker/server connection');
                    fclose(o.connection);
               end
               debug('Delete worker/server connection');
               delete(o.connection);
           end
           if ~o.isWorker
               if isa(o.timer,'timer') && isvalid(o.timer)
                    debug('Stop and delete timer');
                    stop(o.timer);
                    delete(o.timer);
               end
               % Let's close the progressbar after we are sure that no more
               % data will be collected
               progressbar(1.0);
           end
       end
   end
end

% In this function we usually receive the progress of each worker
% This function belongs to the udp connection of the server/main thread and
% is called whenever data from a worker is received. 
% It is also used to log the ip address and port of each worker when they
% connect at the beginning of their execution.
function receiver(h, ~, o)
    [data,count,msg,ip,port] = fread(h, 1, 'ulong');
    if count ~= 2 % error
        debug('Unkown received data from %s:%d with count = %d and fread msg = %s', ip, port, count, msg);
    else
        id = data(1);
        if data(2) == 0 % log in request in worker constructor
            o.workerTable.progress(id) = 0;
            o.workerTable.ip(id) = ip;
            o.workerTable.port(id) = port;
            o.workerTable.connected(id) = true;
            debug('login worker id=%02d with ip:port=%s:%d',id,ip,port);
        else % from worker increment call
            o.workerTable.progress(id) = data(2);
            debug('Set progress for worker id=%02d to %d',id,data(2));
        end
    end
end

% This function is called by the main threads timer to calculate and draw
% the progress bar
% if showWorkerProgress was set to true then the estimated progress of each
% worker thread is displayed (assuming the workload is evenly split)
function draw_progress_bar(~, ~, o)
%     import libUtil.progressbar;
    progressTotal = sum(o.workerTable.progress) / o.totalIterations;
    if progressTotal > o.progressTotalOld
        o.progressTotalOld = progressTotal;
        if(o.showWorkerProgress)
            numWorkers = sum(o.workerTable.connected);
            EstWorkPerWorker = o.totalIterations / numWorkers;
            progWorker = double(o.workerTable.progress) / EstWorkPerWorker;
            progWorkerC = mat2cell(progWorker,ones(1,length(progWorker)));
            progressbar(progressTotal, progWorkerC{:});
        else
            progressbar(progressTotal);
        end
    end
end

% Workers within the parfor loop can sometimes display the commands using
% printf or disp. However, if you start a timer or udp connection and want
% to display anything after an interrupt occured, it is simply impossible
% to print anything. Unfortunately error messages also don't get shown...
% I used this method to just print stuff to a file with the info about
% the current worker/server (main thread). 
function debug(varargin)
%     fid = fopen('E:/tmp/debugParforProgressbar.txt', 'a');
%     t = getCurrentTask(); 
%     if isempty(t)
%         fprintf(fid, 'Server: ');
%     else
%         fprintf(fid, 'Worker ID=%02d: ', t.ID);
%     end
%     fprintf(fid, varargin{:});
%     fprintf(fid, '\n');
%     fclose(fid);
end

function progressbar(varargin)
% Description:
%   progressbar() provides an indication of the progress of some task using
% graphics and text. Calling progressbar repeatedly will update the figure and
% automatically estimate the amount of time remaining.
%   This implementation of progressbar is intended to be extremely simple to use
% while providing a high quality user experience.
%
% Features:
%   - Can add progressbar to existing m-files with a single line of code.
%   - Supports multiple bars in one figure to show progress of nested loops.
%   - Optional labels on bars.
%   - Figure closes automatically when task is complete.
%   - Only one figure can exist so old figures don't clutter the desktop.
%   - Remaining time estimate is accurate even if the figure gets closed.
%   - Minimal execution time. Won't slow down code.
%   - Randomized color. When a programmer gets bored...
%
% Example Function Calls For Single Bar Usage:
%   progressbar               % Initialize/reset
%   progressbar(0)            % Initialize/reset
%   progressbar('Label')      % Initialize/reset and label the bar
%   progressbar(0.5)          % Update
%   progressbar(1)            % Close
%
% Example Function Calls For Multi Bar Usage:
%   progressbar(0, 0)         % Initialize/reset two bars
%   progressbar('A', '')      % Initialize/reset two bars with one label
%   progressbar('', 'B')      % Initialize/reset two bars with one label
%   progressbar('A', 'B')     % Initialize/reset two bars with two labels
%   progressbar(0.3)          % Update 1st bar
%   progressbar(0.3, [])      % Update 1st bar
%   progressbar([], 0.3)      % Update 2nd bar
%   progressbar(0.7, 0.9)     % Update both bars
%   progressbar(1)            % Close
%   progressbar(1, [])        % Close
%   progressbar(1, 0.4)       % Close
%
% Notes:
%   For best results, call progressbar with all zero (or all string) inputs
% before any processing. This sets the proper starting time reference to
% calculate time remaining.
%   Bar color is choosen randomly when the figure is created or reset. Clicking
% the bar will cause a random color change.
%
% Demos:
%     % Single bar
%     m = 500;
%     progressbar % Init single bar
%     for i = 1:m
%       pause(0.01) % Do something important
%       progressbar(i/m) % Update progress bar
%     end
% 
%     % Simple multi bar (update one bar at a time)
%     m = 4;
%     n = 3;
%     p = 100;
%     progressbar(0,0,0) % Init 3 bars
%     for i = 1:m
%         progressbar([],0) % Reset 2nd bar
%         for j = 1:n
%             progressbar([],[],0) % Reset 3rd bar
%             for k = 1:p
%                 pause(0.01) % Do something important
%                 progressbar([],[],k/p) % Update 3rd bar
%             end
%             progressbar([],j/n) % Update 2nd bar
%         end
%         progressbar(i/m) % Update 1st bar
%     end
% 
%     % Fancy multi bar (use labels and update all bars at once)
%     m = 4;
%     n = 3;
%     p = 100;
%     progressbar('Monte Carlo Trials','Simulation','Component') % Init 3 bars
%     for i = 1:m
%         for j = 1:n
%             for k = 1:p
%                 pause(0.01) % Do something important
%                 % Update all bars
%                 frac3 = k/p;
%                 frac2 = ((j-1) + frac3) / n;
%                 frac1 = ((i-1) + frac2) / m;
%                 progressbar(frac1, frac2, frac3)
%             end
%         end
%     end
%
% Author:
%   Steve Hoelzer
%
% Revisions:
% 2002-Feb-27   Created function
% 2002-Mar-19   Updated title text order
% 2002-Apr-11   Use floor instead of round for percentdone
% 2002-Jun-06   Updated for speed using patch (Thanks to waitbar.m)
% 2002-Jun-19   Choose random patch color when a new figure is created
% 2002-Jun-24   Click on bar or axes to choose new random color
% 2002-Jun-27   Calc time left, reset progress bar when fractiondone == 0
% 2002-Jun-28   Remove extraText var, add position var
% 2002-Jul-18   fractiondone input is optional
% 2002-Jul-19   Allow position to specify screen coordinates
% 2002-Jul-22   Clear vars used in color change callback routine
% 2002-Jul-29   Position input is always specified in pixels
% 2002-Sep-09   Change order of title bar text
% 2003-Jun-13   Change 'min' to 'm' because of built in function 'min'
% 2003-Sep-08   Use callback for changing color instead of string
% 2003-Sep-10   Use persistent vars for speed, modify titlebarstr
% 2003-Sep-25   Correct titlebarstr for 0% case
% 2003-Nov-25   Clear all persistent vars when percentdone = 100
% 2004-Jan-22   Cleaner reset process, don't create figure if percentdone = 100
% 2004-Jan-27   Handle incorrect position input
% 2004-Feb-16   Minimum time interval between updates
% 2004-Apr-01   Cleaner process of enforcing minimum time interval
% 2004-Oct-08   Seperate function for timeleftstr, expand to include days
% 2004-Oct-20   Efficient if-else structure for sec2timestr
% 2006-Sep-11   Width is a multiple of height (don't stretch on widescreens)
% 2010-Sep-21   Major overhaul to support multiple bars and add labels
%

persistent progfig progdata lastupdate

% Get inputs
if nargin > 0
    input = varargin;
    ninput = nargin;
else
    % If no inputs, init with a single bar
    input = {0};
    ninput = 1;
end

% If task completed, close figure and clear vars, then exit
if input{1} == 1
    if ishandle(progfig)
        delete(progfig) % Close progress bar
    end
    clear progfig progdata lastupdate % Clear persistent vars
    drawnow
    return
end

% Init reset flag 
resetflag = false;

% Set reset flag if first input is a string
if ischar(input{1})
    resetflag = true;
end

% Set reset flag if all inputs are zero
if input{1} == 0
    % If the quick check above passes, need to check all inputs
    if all([input{:}] == 0) && (length([input{:}]) == ninput)
        resetflag = true;
    end
end

% Set reset flag if more inputs than bars
if ninput > length(progdata)
    resetflag = true;
end

% If reset needed, close figure and forget old data
if resetflag
    if ishandle(progfig)
        delete(progfig) % Close progress bar
    end
    progfig = [];
    progdata = []; % Forget obsolete data
end

% Create new progress bar if needed
if ishandle(progfig)
else % This strange if-else works when progfig is empty (~ishandle() does not)
    
    % Define figure size and axes padding for the single bar case
    height = 0.03;
    width = height * 8;
    hpad = 0.02;
    vpad = 0.25;
    
    % Figure out how many bars to draw
    nbars = max(ninput, length(progdata));
    
    % Adjust figure size and axes padding for number of bars
    heightfactor = (1 - vpad) * nbars + vpad;
    height = height * heightfactor;
    vpad = vpad / heightfactor;
    
    % Initialize progress bar figure
    left = (1 - width) / 2;
    bottom = (1 - height) / 2;
    progfig = figure(...
        'Units', 'normalized',...
        'Position', [left bottom width height],...
        'NumberTitle', 'off',...
        'Resize', 'off',...
        'MenuBar', 'none' );
    
    % Initialize axes, patch, and text for each bar
    left = hpad;
    width = 1 - 2*hpad;
    vpadtotal = vpad * (nbars + 1);
    height = (1 - vpadtotal) / nbars;
    for ndx = 1:nbars
        % Create axes, patch, and text
        bottom = vpad + (vpad + height) * (nbars - ndx);
        progdata(ndx).progaxes = axes( ...
            'Position', [left bottom width height], ...
            'XLim', [0 1], ...
            'YLim', [0 1], ...
            'Box', 'on', ...
            'ytick', [], ...
            'xtick', [] );
        progdata(ndx).progpatch = patch( ...
            'XData', [0 0 0 0], ...
            'YData', [0 0 1 1] );
        progdata(ndx).progtext = text(0.99, 0.5, '', ...
            'HorizontalAlignment', 'Right', ...
            'FontUnits', 'Normalized', ...
            'FontSize', 0.7 );
        progdata(ndx).proglabel = text(0.01, 0.5, '', ...
            'HorizontalAlignment', 'Left', ...
            'FontUnits', 'Normalized', ...
            'FontSize', 0.7 );
        if ischar(input{ndx})
            set(progdata(ndx).proglabel, 'String', input{ndx})
            input{ndx} = 0;
        end
        
        % Set callbacks to change color on mouse click
        set(progdata(ndx).progaxes, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
        set(progdata(ndx).progpatch, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
        set(progdata(ndx).progtext, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
        set(progdata(ndx).proglabel, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
        
        % Pick a random color for this patch
        changecolor([], [], progdata(ndx).progpatch)
        
        % Set starting time reference
        if ~isfield(progdata(ndx), 'starttime') || isempty(progdata(ndx).starttime)
            progdata(ndx).starttime = clock;
        end
    end
    
    % Set time of last update to ensure a redraw
    lastupdate = clock - 1;
    
end

% Process inputs and update state of progdata
for ndx = 1:ninput
    if ~isempty(input{ndx})
        progdata(ndx).fractiondone = input{ndx};
        progdata(ndx).clock = clock;
    end
end

% Enforce a minimum time interval between graphics updates
myclock = clock;
if abs(myclock(6) - lastupdate(6)) < 0.01 % Could use etime() but this is faster
    return
end

% Update progress patch
for ndx = 1:length(progdata)
    set(progdata(ndx).progpatch, 'XData', ...
        [0, progdata(ndx).fractiondone, progdata(ndx).fractiondone, 0])
end

% Update progress text if there is more than one bar
if length(progdata) > 1
    for ndx = 1:length(progdata)
        set(progdata(ndx).progtext, 'String', ...
            sprintf('%1d%%', floor(100*progdata(ndx).fractiondone)))
    end
end

% Update progress figure title bar
if progdata(1).fractiondone > 0
    runtime = etime(progdata(1).clock, progdata(1).starttime);
    timeleft = runtime / progdata(1).fractiondone - runtime;
    timeleftstr = sec2timestr(timeleft);
    titlebarstr = sprintf('%2d%%    %s remaining', ...
        floor(100*progdata(1).fractiondone), timeleftstr);
else
    titlebarstr = ' 0%';
end
set(progfig, 'Name', titlebarstr)

% Force redraw to show changes
drawnow

% Record time of this update
lastupdate = clock;

end
% ------------------------------------------------------------------------------
function changecolor(h, e, progpatch) %#ok<INUSL>
% Change the color of the progress bar patch

% Prevent color from being too dark or too light
colormin = 1.5;
colormax = 2.8;

thiscolor = rand(1, 3);
while (sum(thiscolor) < colormin) || (sum(thiscolor) > colormax)
    thiscolor = rand(1, 3);
end

set(progpatch, 'FaceColor', thiscolor)
end

% ------------------------------------------------------------------------------
function timestr = sec2timestr(sec)
% Convert a time measurement from seconds into a human readable string.

% Convert seconds to other units
w = floor(sec/604800); % Weeks
sec = sec - w*604800;
d = floor(sec/86400); % Days
sec = sec - d*86400;
h = floor(sec/3600); % Hours
sec = sec - h*3600;
m = floor(sec/60); % Minutes
sec = sec - m*60;
s = floor(sec); % Seconds

% Create time string
if w > 0
    if w > 9
        timestr = sprintf('%d week', w);
    else
        timestr = sprintf('%d week, %d day', w, d);
    end
elseif d > 0
    if d > 9
        timestr = sprintf('%d day', d);
    else
        timestr = sprintf('%d day, %d hr', d, h);
    end
elseif h > 0
    if h > 9
        timestr = sprintf('%d hr', h);
    else
        timestr = sprintf('%d hr, %d min', h, m);
    end
elseif m > 0
    if m > 9
        timestr = sprintf('%d min', m);
    else
        timestr = sprintf('%d min, %d sec', m, s);
    end
else
    timestr = sprintf('%d sec', s);
end
end
