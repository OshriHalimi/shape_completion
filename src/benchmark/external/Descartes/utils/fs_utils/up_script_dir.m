function path = up_script_dir(n,join)
    % If n = 0: Returns the directory of the caller code 
    % If n>0: Returns the nth directory up from the caller code dir
    % join is a string array meant for usage with the fullfile command
    % Constraint: n>=0, isinteger(n) == true. If n is too big - empty array returned  
    assert(n>=0 && floor(n)==n == true); 
    ST = dbstack;
    path = fileparts(which(ST(2).file));
    for i=1:n
        path = fileparts(path); 
    end
    if nargin > 1
        path = fullfile(path, join); 
    end
end