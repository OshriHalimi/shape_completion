function [vnew, fnew, meanedge, stdev]=remesher(V, F, edgelength, iterations)
% INPUT
% -V: vertices of mesh; n*3 array of xyz coordinates
% -F: faces of  mesh; n*3 array
% -edgelength:  edgelength aimed for
% -iterations: nr of iterations, 5 to 10 suffice
% 

% OUTPUT
% -vnew: remeshed vertices list: n*3 array of xyz coordinates
% -fnew: remeshed list of faces of  mesh; n*3 array
% -meanedge: average edgelenth obtained
% -stdev: stdev of edgelengths

% EXAMPLE
%     load testdatafemur
%     [vnew, fnew, meanedge, stdev]=remesher(vnew, fnew, 3, 5);

%clean up patch
[vnew, fnew]=cleanpatch(V, F);
voriginal=vnew;
foriginal=fnew;
[vnew,fnew] = subdividelarge( vnew, fnew,edgelength,voriginal,foriginal );
voriginal=vnew;
foriginal=fnew;

for i=1:iterations
    
[vnew, fnew,temp] = edgecollaps( vnew, fnew, edgelength ,voriginal,foriginal);
[vnew, fnew, temp] = removebadtriangles( vnew, fnew,voriginal,foriginal);
[vnew,fnew] = subdividelarge( vnew, fnew,0 ,voriginal,foriginal);
[vnew, fnew, temp] = removebadtriangles( vnew, fnew,voriginal,foriginal);

disp(['Iteration:' num2str(i) '  Output mesh: ' num2str(size(fnew,1)) ' triangles, ' ... 
    num2str(size(vnew,1))  ' vertices.']);
end


meanedge=temp(:,1);
stdev=temp(:,2);

