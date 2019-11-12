function [v,f,c] = read_off(file)

c = []; 
fid = fopen(file,'r');
if fid==-1 ;error('Non-existant file %s',file); end

% PARSE NAME 
header = fgets(fid);   
is_off = strcmp(header(1:3), 'OFF');
if ~is_off && ~strcmp(header(1:4), 'COFF') 
    error('The file %s is not a valid OFF/COFF file',file); 
end

% PARSE HEADER 
fv_cnts = fgets(fid);
[tok,fv_cnts] = strtok(fv_cnts); 
Nv = str2num(tok);
while isempty(tok) || isempty(Nv)
    fv_cnts = fgets(fid);
    [tok,fv_cnts] = strtok(fv_cnts); 
    Nv = str2num(tok);
end
[tok,~] = strtok(fv_cnts); Nf = str2num(tok);

% PARSE VERTICES
if is_off
    [A,cnt] = fscanf(fid,'%f %f %f', 3*Nv);
    if cnt~=3*Nv
        error('Invalid VERTEX format in off file');
    end
    A = reshape(A, 3, cnt/3);
    v = A;
else % is COFF 
    [A,cnt] = fscanf(fid,'%f %f %f %d %d %d %d', 7*Nv);
    if cnt~=7*Nv
        error('Invalid VERTEX/COLOR format in coff file');
    end
    A = reshape(A, 7, cnt/7);
    v = A(1:3,:); 
    c = (A(4:6,:)./255).'; % Lose the Alpha Channel. TODO: Fix size on all other files
end

[A,cnt] = fscanf(fid,'%d %d %d %d\n', 4*Nf);
if cnt~=4*Nf
    error('Invalid FACE format in off file');
end
A = reshape(A, 4, cnt/4);
f = A(2:4,:)+1; % Face indices start from 0, and not 1

fclose(fid);

