import numpy as np
import matplotlib.pyplot as plt
import ipdb

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

def eIAF(spikes,threshold,tau):
    """ generate an intergrate and fire spike train. """

    filt = np.zeros_like(spikes)
    out_filt = np.zeros_like(spikes)
    out_spikes = np.zeros_like(spikes)
    
    for i in np.arange(1,len(spikes)):
        filt[i] = tau* filt[i-1] + spikes[i] 
        if filt[i] > threshold:
            filt[i] = 0
            out_spikes[i] = 1
    
    for i in np.arange(1,len(spikes)):
      out_filt[i] = tau* out_filt[i-1] + out_spikes[i] 
           

    return (out_spikes,out_filt,filt)

def LIF(spikes,threshold,tau_n, tau_s= 0.05, w=0.4,synapse=True):

    v = np.zeros_like(spikes)
    s = np.zeros_like(spikes)
    out_spikes = np.zeros_like(spikes)

    print len(v)
    # For every timestep
    for i in np.arange(v.shape[0]):
        # update synape
        if synapse:
            # With a exponential synapse, the task is currently unlearnable
            s[i] = ((s[i-1]) /tau_s) + w* spikes[i] 
            v[i] = ((v[i-1]) /tau_n) + s[i-1] 
        else:
            # Without an exponential synapse, the task is easily learned
            v[i] = ((v[i-1]) /tau_n) + spikes[i-1] 


        if v[i] > threshold: 
            v[i] = 0
            out_spikes[i] = 1

    return (out_spikes,v,s)



if __name__ == "__main__":

    tau = 0.8
    spikes = np.zeros(100)
    spikes[::11] = 1    
    spikes[::13] = 1

    spike,trace = eIAF(spikes,1,tau)
    plt.plot(spike); plt.plot(trace); plt.show()
