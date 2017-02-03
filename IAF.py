import numpy as np

def IAF(current,threshold):
    """ generate an intergrate and fire spike train. """
    
    trace = np.zeros(current.shape)
    spikes = np.zeros(current.shape)

    for i in np.arange(len(current))-1:
        i = i+ 1
        trace[i] = trace[i-1] + current[i]
        if trace[i] > threshold:
            spikes[i] = 1
            trace[i] = 0

    spike_string = ''
    for i in np.arange(len(current))-1:
        spike_string = spike_string + str(spikes[i])
    return (spikes,trace,spike_string)

