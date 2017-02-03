import matplotlib.pyplot as plt
import numpy as np
from neuron import LIF  

in_spikes = 1.0*(np.random.random((1000,1))<0.05)
print in_spikes.shape
x,v,s = LIF(in_spikes,1,1/0.8,1/0.4,0.7,True) 

plt.plot(x); plt.show()
