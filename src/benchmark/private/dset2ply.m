function dset2ply(dset)

N = size(dset.fvs,1);
for i=2:N
    Mesh(dset.fvs{i,1}.vertices,dset.fvs{i,1}.faces).export_as(sprintf('res_%d.ply',i-1));
    Mesh(dset.fvs{i,2}.vertices,dset.fvs{i,2}.faces).export_as(sprintf('gt_%d.ply',i-1));
    Mesh(dset.fvs{i,3}.vertices,dset.fvs{i,3}.faces).export_as(sprintf('part_%d.ply',i-1));
    Mesh(dset.fvs{i,4}.vertices,dset.fvs{i,4}.faces).export_as(sprintf('tp_%d.ply',i-1));
end
banner('Export finished');

