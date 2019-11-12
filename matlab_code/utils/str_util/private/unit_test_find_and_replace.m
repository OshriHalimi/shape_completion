classdef unit_test_find_and_replace < matlab.unittest.TestCase

% Unit test for find_and_replace
%
% This class executes several examples to test the operation of the 
% find_and_replace function. It uses the MATLAB(R) Unit Test framework,
% which was introduced in MATLAB R2013a.
%
% Tucker McClure
% Copyright 2013, The MathWorks, Inc.
    
    properties
        
        % We'll consistently write to and read from this file, and we'll
        % delete it when the testing is done.
        file_name;
        path_name;
        
    end
    
    methods (Test)
        
        % Check that we leave different cases alone.
        function test_case_sensitivity(tc)
            
            % Set original and expected strings.
            original = sprintf('This and that.\nAnd this.');
            expected = sprintf('This and that.\nAnd those.');
            
            % Write out the test file.
            tc.write_file(original);
            
            % Update it.
            find_and_replace(tc.file_name, 'this', 'those');
            
            % Read the result.
            result = tc.read_file();
            
            % See if they match.
            tc.verifyEqual(expected, result);
            
        end
        
        % Check that we find words at the very beginning.
        function test_word_at_start(tc)
            
            % Set original and expected strings.
            original = sprintf('This and that.\nAnd this.');
            expected = sprintf('Those and that.\nAnd this.');
            
            % Write out the test file.
            tc.write_file(original);
            
            % Update it.
            find_and_replace(tc.file_name, 'This', 'Those');
            
            % Read the result.
            result = tc.read_file();
            
            % See if they match.
            tc.verifyEqual(expected, result);
            
        end
        
        % Check for variable name replacement.
        function test_variable_names(tc)
            
            % Set original and expected strings.
            original = sprintf('my_var = 3;\nmy_var_2 = my_var + my_var');
            expected = sprintf('foo = 3;\nbar = foo + foo');
            
            % Write out the test file.
            tc.write_file(original);
            
            % Update it.
            find_and_replace(tc.file_name, '\<my_var\>', 'foo');
            find_and_replace(tc.file_name, '\<my_var_2\>', 'bar');
            
            % Read the result.
            result = tc.read_file();
            
            % See if they match.
            tc.verifyEqual(expected, result);
            
        end
        
        % Check for other regular expressions.
        function test_regular_expressions(tc)
            
            % Set original and expected strings.
            original = 'I walk up, they walked up, we are walking up.';
            expected = 'I ascend, they ascended, we are ascending.';
            
            % Write out the test file.
            tc.write_file(original);
            
            % Update it.
            find_and_replace(tc.file_name, 'walk(\w*) up', 'ascend$1');
            
            % Read the result.
            result = tc.read_file();
            
            % See if they match.
            tc.verifyEqual(expected, result);
            
        end
        
        % Check for other regular expressions.
        function test_function_replacement(tc)
            
            % Set original and expected strings.
            original = 'sqrt(3 * a)';
            expected = 'my_sqrt(3 * a)';
                
            % Write out the test file.
            tc.write_file(original);
            
            % Update it.
            find_and_replace(tc.file_name, ...
                             'sqrt\((.*?)\)', 'my_sqrt\($1\)');
            
            % Read the result.
            result = tc.read_file();
            
            % See if they match.
            tc.verifyEqual(expected, result);
            
        end
        
        % Test multiple files with struct input (such as from dir).
        function test_multiple_files_struct(tc)
            
            % Set original and expected strings.
            original = 'sqrt(3 * a)';
            expected = 'my_sqrt(3 * a)';
                
            % Write out the test files.
            tc.write_file(original, [tc.path_name '.file_1.test']);
            tc.write_file(original, [tc.path_name '.file_2.test']);
            
            % Update them.
            find_and_replace(dir([tc.path_name '*.test']), ...
                             'sqrt\((.*?)\)', 'my_sqrt\($1\)');
            
            % Read the results.
            result_1 = tc.read_file([tc.path_name '.file_1.test']);
            result_2 = tc.read_file([tc.path_name '.file_2.test']);
            
            % See if they match.
            tc.verifyEqual(expected, result_1);
            tc.verifyEqual(expected, result_2);
            
            % Delete the old files.
            delete([tc.path_name '.file_1.test']);
            delete([tc.path_name '.file_2.test']);
            
        end
        
        % Test multiple files with cell array input.
        function test_multiple_files_cell(tc)
            
            % Set original and expected strings.
            original = 'sqrt(3 * a)';
            expected = 'my_sqrt(3 * a)';
                
            % Write out the test files.
            tc.write_file(original, [tc.path_name '.file_1.test']);
            tc.write_file(original, [tc.path_name '.file_2.test']);
            
            % Update them.
            files = {[tc.path_name '.file_1.test']; ...
                     [tc.path_name '.file_2.test']};
            find_and_replace(files, 'sqrt\((.*?)\)', 'my_sqrt\($1\)');
            
            % Read the results.
            result_1 = tc.read_file([tc.path_name '.file_1.test']);
            result_2 = tc.read_file([tc.path_name '.file_2.test']);
            
            % See if they match.
            tc.verifyEqual(expected, result_1);
            tc.verifyEqual(expected, result_2);
            
            % Delete the old files.
            delete([tc.path_name '.file_1.test']);
            delete([tc.path_name '.file_2.test']);
            
        end
        
        % Check for correct errors when no file exists or is given
        % incorrectly.
        function test_incorrect_file(tc)
            
            % Try to find and replace in a bogus file, and see that it
            % produces the expected error.
            f = @() find_and_replace([tc.path_name 'bogus'], 'a', 'b');
            tc.verifyError(f, 'find_and_replace:no_file');
            
            % Try to find and replace in a bogus file.
            f = @() find_and_replace(-1, 'a', 'b');
            tc.verifyError(f, 'find_and_replace:invalid_inputs');
            
        end
        
    end
    
    methods (TestMethodSetup)
        
        % When we start up, find this class's location and store the
        % temporary files there.
        function set_file_name(ut)
            p = mfilename('fullpath');
            p = p(1:find(p == filesep, 1, 'last'));
            ut.path_name = p;
            ut.file_name = [p '.unit_test_find_and_replace.txt'];
        end
        
    end
    
    methods (TestMethodTeardown)
        
        % When tests are done, delete the temporary file.
        function delete_file(ut)
            if exist(ut.file_name, 'file')
                delete(ut.file_name);
            end
        end
        
    end

    methods
                
        % Write out a string to a file with the given name.
        function write_file(ut, string, name)
            if nargin == 2
                name = ut.file_name;
            end
            fid = fopen(name, 'w');
            fprintf(fid, string);
            fclose(fid);
        end
        
        % Read in a whole file as a long string of chars.
        function string = read_file(ut, name)
            if nargin == 1
                name = ut.file_name;
            end
            fid = fopen(name);
            string = fread(fid, '*char')';
            fclose(fid);
        end
        
    end
    
end
