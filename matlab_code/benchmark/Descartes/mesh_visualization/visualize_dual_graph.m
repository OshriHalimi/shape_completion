function visualize_dual_graph(M)
M.plt.title = 'Dual Graph'; 
M.ezvisualize();
M = M.add_fedge_set();
add_edge_visualization(M,M.fe,1,'b'); 
add_edge_visualization(M,M.e_feal,0,'c'); 
end