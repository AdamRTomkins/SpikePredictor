from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

def poisson_neuron(f=10,steps=1000):

    P = PoissonGroup(1, np.arange(1)*Hz + f*Hz)
    G = NeuronGroup(1, 'dv/dt = -v / (100*ms) : 1',threshold='v>0.8',reset='v = 0')
    spikemon = SpikeMonitor(G)
    pspikemon = SpikeMonitor(P) 
    S = Synapses(P, G, pre='v+=0.7')
    S.connect(j='i')

    M = StateMonitor(G, 'v', record=True)
    run(steps*ms)
    #plt.plot(M.v[0]); plt.show()

    return (np.array(pspikemon.t[:]),np.array(spikemon.t[:]),M.v[0])


if __name__ == "__main__":    
    poisson_neuron(30)


