function [] = banner(string)

FULL_HEADER_LENGTH = 80;
if ~exist('string','var')
    string ='';
else
    string = [' ' uplw(string) ' ' ];
end

str_len = length(string);
header_length_actual = FULL_HEADER_LENGTH - str_len;
each_side_len = floor((header_length_actual)/2);
disp([repmat('=',1,each_side_len) string repmat('=',1,header_length_actual - each_side_len)])
% fprintf('%s\n\n', repmat('.',1,FULL_HEADER_LENGTH))
end

