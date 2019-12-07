function sup = supported_ext(short)
if exist('short','var') && short
    sup = {'off','coff','ply','smf','wrl', 'obj','tet','mat','stl'};
else
    sup = {'*.off','*.coff','*.ply','*.smf','*.wrl', '*.obj','*.tet','*.mat','*.stl'};
end
end