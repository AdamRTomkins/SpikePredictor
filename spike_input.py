""" A script to use our spike trainer to learn a current -> spiking relationship """

from spike_trainer_2 import spike_trainer
from neuron import IAF, LIF

#from brian_neurons import poisson_neuron

import numpy as np
from matplotlib import pyplot as plt

import random
import ipdb



def  exp_LeakyIAF_input(nb_epoch,num_steps=10000,test_steps=1000,exp_name='exp_LeakyIAF_input',load_model=True,load_weights=True):
    from brian_models import LIF
    

    in_trace, v_trace, spikes_pre, spikes_post = LIF(num_steps/10000.)    
    in_trace_test, v_trace_test, pikes_pre_test, spikes_post_test = LIF(test_steps/10000.)    

    

    st = spike_trainer(filter(spikes_pre),filter(spikes_post),exp_name=exp_name,nb_epoch = nb_epoch,load_weights=load_weights,load_model=load_model)  
    st.fit_model()
    predicted_spikes = st.compute(filter(in_trace_test))
    print "reproduced " + str(sum(predicted_spikes)) + " of " + str(sum(spikes_post_test[100:]))
    
    plt.plot(spikes_post_test[100:]); plt.plot(predicted_spikes); plt.show()
    #plt.plot(out_filter_test[100:]); plt.plot(predicted_spikes); plt.show()
    ipdb.set_trace()


def  exp_true_lif_input(nb_epoch,num_steps=10000,test_steps=1000,exp_name='exp_true_input',load_model=True,load_weights=True):

    spikes_pre          = 1.0*(np.random.random((num_steps,1))<0.1)
    spikes_post,v,s     =  LIF(spikes_pre,1,1/0.8,1/0.4,0.7,False)

    spikes_pre_test     = 1.0*(np.random.random((test_steps,1))<0.1)
    spikes_post_test,v,s = LIF(spikes_pre_test,1,1/0.8,1/0.4,0.7,False)


    ipdb.set_trace()

    st = spike_trainer(filter(spikes_pre),filter(spikes_post),exp_name=exp_name,nb_epoch = nb_epoch,load_weights=load_weights,load_model=load_model)  

    st.fit_model()

    predicted_spikes = st.compute(filter(spikes_pre_test))
    plt.plot(predicted_spikes); plt.plot(spikes_post_test[100:]); plt.show()

    ipdb.set_trace()
    
def filter(a): 
    f = np.zeros_like(a)
    num_steps = len(a)
    print num_steps
    for i in np.arange(1,num_steps):
        f[i] = 0.80*f[i-1] + a[i]
    return f/np.max(f)

if __name__ == "__main__":
   #exp_LeakyIAF_input(nb_epoch = 5,num_steps = 100000, exp_name = 'exp_LeakyIAF_filter_input', load_model=True, load_weights=True)
   exp_true_lif_input(nb_epoch = 10,num_steps = 100000, exp_name = 'exp_lif_dense_input', load_model=True, load_weights=True)




