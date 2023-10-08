import numpy as np

def baseline_removal(data,baseline_points):
    # data: 4D array [num_targets,num_channels,num_samples,num_blocks]
    # baseline_points: baseline_samplepoints
    # data_out: 4D array [num_targets,num_channels,num_samples,num_blocks]
    
    data_out = data - np.mean(data[:,:,baseline_points,:],axis=2,keepdims=True)
    return data_out