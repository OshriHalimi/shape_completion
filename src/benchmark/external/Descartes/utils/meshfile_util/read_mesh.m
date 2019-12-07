function [M,f] = read_mesh(file,varargin)
% READ_MESH - read data from standard mesh files
%   Supported file extensions are: .off, .ply, .wrl, .obj, .m, .gim,.mat

if nargin==0 % Let user choose a file
    [f, pathname] = uigetfile(supported_ext(),'Choose a mesh file');
    file = [pathname,f];
end
[~,name,ext] = fileparts(file);
ext = lower(ext); 
if isempty(ext) || strcmp(ext,'.m') % Mesh Zoo
    switch name % Common naming mistakes & redirections
        case 'bucky'
            name = 'buckyball';
        case 'cylinder' 
            name = 'cyl';
    end
    if nargin>1
        M = feval(name, varargin{:}); % Treat filename as function
    else
        M = feval(name);
    end
else
    % c = []; 
    switch ext
        case {'.off','.coff'}
            [v,f,~] = read_off(file);
        case '.stl'
            [v,f] = read_stl(file);
        case '.ply'
            [v,f] = read_ply(file);
        case '.smf'
            [v,f] = read_smf(file);
        case '.wrl'
            [v,f] = read_wrl(file);
        case '.obj'
%             try
                [v,f] = read_obj_fast(file);
%             catch
%                 OBJ = read_obj(file);
%                 v=OBJ.vertices;
%                 f=OBJ.objects.data.vertices;
%             end
        case '.tet'
            [v,f] = read_tet(file);
        case '.mat'
            S = load(file);
            if isfield(S,'M')
                M = S.M; 
                % TODO: Remove this!
                M.path = which(file); 
                [~,M.file_name,~] = fileparts(M.path);
                if nargout == 2
                    f = M.f; 
                    M = M.v; 
                end
                return;
            end
            % Possible Dereferencing 
            if isfield(S,'surface')
                S = S.surface;
            end
            if isfield(S,'shape')
                S = S.shape;
            end
            
            % Try loading facets
            if isfield(S,'TRIV')
                f = S.TRIV;
            elseif isfield(S,'f')
                f = S.f;
            elseif isfield(S,'faces')
                f = S.faces; 
            else
                error('Could not find faces in mat file');
            end
            % Try loading vertices
            if isfield(S,'v')
                v = S.v;
            elseif isfield(S,'X')
                v = [S.X,S.Y,S.Z];
            elseif isfield(S,'verts')
                v = S.verts; 
            else
                error('Could not find vertices in mat file');
            end
        otherwise
            error('Invalid mesh-file extension');
    end
    if nargin >1 % Treat additional input as name override
        name = varargin{1};
    else
        % Try Testset Naming convention:  
        [s,~] = regexp(name,'(.*)_\d+k[DbS]*', 'tokens','match');
        if ~isempty(s)
            s = s{1}; name = s{1};
        else
            [s,~] = regexp(name,'(.*)_mini[DbS]*', 'tokens','match');
            if ~isempty(s)
                s = s{1}; name = s{1};
            end
        end
        % Worst case - use the original name
    end
    name = uplw(name); 
    filepath = which(file); 
    if isempty(filepath)
        filepath = file; 
    end
    try
        M = Mesh(v,f,name,filepath);
%         if ~isempty(c)
%             cprintf('Blue','Note: Coloring found in mesh file\n'); 
%         end
    catch ME
        warning('Failed constructor at %s',file); 
        rethrow(ME); 
    end
    if nargout == 2
        f = M.f; 
        M = M.v; 
    end
%     if exist('v_nrmls','var')
%         M.vn = v_nrmls;
%     end
end



% ext = lower(ext);
% assert(~isempty(validatestring(ext,EXT)),'Invalid extension');
