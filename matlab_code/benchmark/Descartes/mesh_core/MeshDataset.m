classdef MeshDataset
    properties
        path % Origin
        name % Dataset name
        cache_path % Path to cache
        
        ms % Mesh
        N % Number of meshes
        Nc % Number of classes
        cls % Class categorical array
        
        c % Spares - Dump of all extra info
        
        is_manifold
        is_complete
        is_watertight
    end
    methods
%-------------------------------------------------------------------------%
%                                 Constructor
%-------------------------------------------------------------------------%
        function [ds]= MeshDataset(name,force)
            ds.name = lower(name);
            ds.path=name2path(ds.name);
            ds.cache_path = fullfile(ds.path,sprintf('%s_ds.mat',ds.name));
            
            if ~(exist('force','var') && force) && isfile(ds.cache_path)
                load(ds.cache_path,'ds');
            else % Load ALL meshes found (TODO - sparse it)
                mfp = findfiles(ds.path,supported_ext());
                mfp = mfp(~ismember(mfp,ds.cache_path)); % Remove cache file
                N_mfp = numel(mfp); ds.ms = cell(N_mfp,1);  progressbar;
                for i=1:N_mfp
                    ds.ms{i} = read_mesh(mfp{i}); progressbar(i/N_mfp);
                end
                ds.N = numel(ds.ms);
                ds = parse_classes(ds,mfp);
                ds = ds.validate();
                ds.cache();
            end
            fprintf('Successfully loaded dataset %s with %d meshes and %d classes\n',upper(name),ds.N,ds.Nc);
        end
%-------------------------------------------------------------------------%
%                             Global Ops & Info
%-------------------------------------------------------------------------%
        function show(ds,id,scramble)
            mo.dset_name = usprintf('%s Dataset',ds.name);
            mo.disp_func = @ezvisualize;
            mo.tight = [0.02,0.02];
            if exist('scramble','var'); mo.perm = scramble; end
            if exist('id','var') && ~isempty(id); ds = ds.subset(id); end
            mesh_montage(ds.ms,mo)
        end
        function export_to_format(ds,dirname,format,face_cnt)
            if ~exist('face_cnt','var'); face_cnt = []; end
            [status,~,~] = mkdir(dirname); assert(status);
            assert(any(strcmp(supported_ext(1),format)),'Invalid format');
            progressbar;
            for i=1:ds.N
                M = ds.ms{i};
                if ~isempty(face_cnt)
                    M = qslim(M,face_cnt); 
                end
                [~,fname,~] = fileparts(M.file_name);
                fname = [fname, '.' , format] ;
                M.export_as(fullfile(dirname,fname));
                progressbar(i/ds.N);
            end
            fprintf('Succesfully exported dataset in %s format to:\n\t%s\n',format,dirname);
        end
        function summary(ds)
            if isempty(ds.cls)
                fprintf('Dataset hold %s meshes, with no class identification\n',ds.N);
            else
                summary(ds.cls);
            end
        end
        function [c] = classes(ds)
            if isempty(ds.cls)
                c = [];
            else
                c = categories(ds.cls);
            end
        end
        function [c] = counts(ds)
            if isempty(ds.cls)
                c = {};
            else
                c = [categories(ds.cls),num2cell(countcats(ds.cls).')];
            end
        end
%-------------------------------------------------------------------------%
%                                   Mutators
%-------------------------------------------------------------------------%
        function ds = validate(ds)
            mani = false(ds.N,1); clsed = false(ds.N,1); cmplt = false(ds.N,1);
            for i=1:ds.N
                [ds.ms{i},sm,mani(i),clsed(i),cmplt(i)] = mesh_status(ds.ms{i});
                fprintf('%d. %s\n',i,sm);
            end
            ds.is_manifold = mani;
            ds.is_watertight = clsed;
            ds.is_complete = cmplt;
        end
        function [ds] = kset(ds,k)
            assert(k>0,'Invalid k value'); 
            maxk = min(countcats(ds.cls)); 
            assert(maxk >= k, 'Value specified for k(%d) is too high - not enough members from each class (%d)',k,maxk);
            [cls_id] = grp2idx(ds.cls);
            curr_id = 0;  
            for i=1:length(cls_id)
                if cls_id(i) == curr_id % still in progress 
                    if cnt < k
                        cnt = cnt+1; 
                    else
                        cls_id(i) = 0; 
                    end
                else 
                    curr_id = cls_id(i); 
                    cnt = 1; 
                end
            end
            ds = ds.subset(cls_id > 0); 
        end
        function [ds] = subset(ds,inds)
            if isnumeric(inds) || islogical(inds)
                if length(inds)==1 && inds < 1 % Address as fraction
                    inds = randperm(ds.N,round(inds*ds.N));
                end
                % Already ind
            elseif ischar(inds) % Class name
                inds = (ds.cls == inds); % May fail if ds.cls is empty
            else % Assume iscell
                inds = ismember(ds.cls,inds);
            end
            oldN  = ds.N; 
            % Truncate ms
            ds.ms = ds.ms(inds);
            ds.N = numel(ds.ms);
            assert(ds.N > 0, 'No figures left in dataset');
            % Trucnate classes
            if ~isempty(ds.cls)
                ds.cls = ds.cls(inds);
                ds.cls = removecats(ds.cls);
                ds.cls = ds.cls(~isundefined(ds.cls));
                ds.Nc = numel(categories(ds.cls));
            end
            % Truncate other members
            ds = trunc_struct(ds,oldN,inds); 
        end
        function [ds] = remove(ds,inds)
            if isnumeric(inds) || islogical(inds)
                keepers = true(ds.N,1);
                keepers(inds) = false;
            else
                keepers = ~ismember(ds.cls,inds); % May fail if d.cls is empty
            end
            ds = ds.subset(keepers);
        end
        function [ds] = filter(ds,manifold,complete,watertight)
            if ~exist('manifold','var'); manifold =1; end
            if ~exist('complete','var'); complete =1; end
            if ~exist('watertight','var'); watertight =0; end
            filter = true(ds.N,1);
            if manifold; filter = filter & ds.is_manifold; end
            if complete; filter = filter & ds.is_complete; end
            if watertight; filter = filter & ds.is_watertight; end
            ds = ds.subset(filter);
            removed = numel(filter) - sum(filter);
            if removed
                fprintf('Filtered %d meshes\n',removed);
            end
        end
        function [M] = get_mesh(ds,id)
            M = [];
            id = lower(id);
            found = false; 
            for i=1:ds.N
                currM = ds.ms{i};
                if strcmpi(currM.name,id) || strcmpi(currM.file_name,id)
                    M = currM; found = true; break;
                end
            end
            assert(found,sprintf('Could not find shape %s in dataset',id)); 
        end
%-------------------------------------------------------------------------%
%                                 Mesh Mutators
%-------------------------------------------------------------------------%
        function [ds] = set_field(ds,name,val)
            fields = strsplit(name,'.');
            for i=1:ds.N
                ds.ms{i} = setfield(ds.ms{i},fields{:},val);
            end
        end
        function [] = print_field(ds,fname)
            fields = strsplit(fname,'.');
            fprintf('Value for %s:\n',fname); 
            for i=1:ds.N
                fprintf('\t%d: %s : %g\n',i,ds.ms{i}.name,getfield(ds.ms{i},fields{:}));
            end
        end
%-------------------------------------------------------------------------%
%                                 Caching
%-------------------------------------------------------------------------%
        function cache(ds)
            save(ds.cache_path,'ds','-v7.3');
            fprintf('Successfully cached dataset at:\n\t%s\n',ds.cache_path);
        end
        function [ds] = clean(ds)
            ds.c = []; 
        end
%-------------------------------------------------------------------------%
%                                   Static
%-------------------------------------------------------------------------%
    end
    methods (Static)
        function [path] = datasets_path()
            persistent dataset_dir_path
            if isempty(dataset_dir_path)
                dataset_dir_path = up_script_dir(2,'datasets');
            end
            path = dataset_dir_path;
        end
        function [dataset_names] = which()
            dataset_names = list_file_names(MeshDataset.datasets_path());
            assert(~isempty(dataset_names),'No datasets found');
            if nargout == 0
                print_list(dataset_names,'Mesh Datasets:')
            end
        end
    end
end

%-------------------------------------------------------------------------%
%                              Private
%-------------------------------------------------------------------------%
function ds_tgt_dir=name2path(name)
ds_dir = MeshDataset.datasets_path();
ds_names = MeshDataset.which();
idx = find(ismember(lower(ds_names),lower(name)), 1);
assert(~isempty(idx),'Did not find dataset %s',name);
ds_tgt_dir = fullfile(ds_dir,ds_names{idx});
end

function ds = parse_classes(ds,mfp)
try %TODO - Make this smarter.
    [~,names,~] = cellfun(@fileparts,mfp,'un',0);
    r = '^([a-zA-Z_\-\.]+)\d*';
    [tokens,~] = regexp(names,r, 'tokens','match');
    tokens = [tokens{:}];tokens = categorical([tokens{:}]);
    ds.cls = tokens; % Categorical array
    ds.Nc = numel(categories(ds.cls));
catch
    warning('Could not parse class names');
    ds.Nc = 0; 
end
end
