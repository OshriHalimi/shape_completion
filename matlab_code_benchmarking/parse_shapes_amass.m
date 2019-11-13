function [subjectIDfull, subjectIDpart, poseIDfull, poseIDpart, projectionID] = parse_shapes_amass(filename, same_subject)
    tokens = split(filename,{'_','.'});
    if same_subject == true
        subjectIDfull = tokens{2};
        subjectIDpart = tokens{2};
        
        poseIDfull = tokens{4};
        poseIDpart = tokens{6};
        
        projectionID = tokens{8};
    else
        subjectIDfull = tokens{2};
        subjectIDpart = tokens{4};
        
        poseIDfull = tokens{6};
        poseIDpart = tokens{8};
        
        projectionID = tokens{10};
    end
end

