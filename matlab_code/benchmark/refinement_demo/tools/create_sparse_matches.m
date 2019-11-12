function [P,source_dsmpl_to_full,target_sparse_matches] = ...
        create_sparse_matches(source_dsmpl,target_dsmpl,source_full,target_full,matches_dsmpl)
%Input:
%source_dsmpl - downdampled source model
%target_dsmpl - downsampled target model
%source_full - full resolution source model. ASSUMED TO BE CLEANED OF OUTLIERS!!!
%target_full - full resolution target model. ASSUMED TO BE CLEANED OF OUTLIERS!!!
%matches_dsmpl - vector of matched from downdampled source model to downsampled target model
%matches_dsmpl(i) maps the i'th vertes of source_dsmpl to the
%matches_dsmpl(i)'th vertex of target_dsmpl
%
%Output:
%P - sparse matrix of matches, dimensions are V_target X V_source
%(number of vertices)
%
%(C) Oshri Halimi 2018

[source_dsmpl_to_full, err] = ...
    knnsearch(source_full.VERT,source_dsmpl.VERT, 'NSMethod', 'exhaustive','K',1);
[target_dsmpl_to_full, err] = ...
    knnsearch(target_full.VERT,target_dsmpl.VERT, 'NSMethod', 'exhaustive','K',1);
target_sparse_matches = target_dsmpl_to_full(matches_dsmpl);
P = sparse(source_dsmpl_to_full,target_sparse_matches,ones(1,source_dsmpl.n),source_full.n,target_full.n);
end

