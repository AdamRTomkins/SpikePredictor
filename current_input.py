""" A script to use our spike trainer to learn a current -> spiking relationship """

from spike_trainer_1 import spike_trainer
from neuron import IAF

#from brian_neurons import poisson_neuron

import numpy as np
from matplotlib import pyplot as plt

import random
import ipdb

def poisson_input(nb_epoch=7,num_steps=20000):
    
    hz = 60
    (pn,sn,v) = poisson_neuron(hz,num_steps)
    
    # go from spike times to zeros
    in_spikes = np.zeros((num_steps,1))
    out_spikes = np.zeros((num_steps,1))


    for s in sn:
       out_spikes[int((s*1000)) ] = 1
    
    for s in pn:
       in_spikes[int((s*1000)) ] = 1    

    st = spike_trainer(in_spikes,out_spikes,nb_epoch=nb_epoch)  
    st.fit_model()
    """ 
    num_steps = 500
    (pn,sn,v) = poisson_neuron(hz,num_steps)
    
    in_spikes = np.zeros((num_steps,1))
    out_spikes = np.zeros((num_steps,1))

    for s in sn:
       out_spikes[int((s*1000)) ] = 1
    
    for s in pn:
       in_spikes[int((s*1000)) ] = 1    
     
    """
    predicted_spikes = np.array(st.compute(in_spikes[1000:2000]))

    print "Ratio = " + str( np.sum(predicted_spikes) / np.sum(out_spikes[1100:2000]))

    plt.plot(out_spikes[1100:2000]); plt.plot(-predicted_spikes);plt.show()
    plt.plot(out_spikes[1100:2000]); plt.plot(-in_spikes[1100:2000]); plt.show()

def exp_constant_input(nb_epoch=7,num_steps=10000,test_steps=1000,exp_name='exp_constant_input',load_model=True,load_weights=True):

    #Training
    current = np.ones((num_steps,1)) * 0.1

    threshold = 1

    (spikes,trace,string) = IAF(current,threshold)
    st = spike_trainer(trace,spikes,nb_epoch=nb_epoch,exp_name=exp_name,load_model=load_model,load_weights=load_weights)  
    st.fit_model()

    test_levels = [0.1]
    results = [{} for l in test_levels]
    for i, level in enumerate(test_levels):
        # Prediction
        print "Predict at " + str(level)
        current = np.ones((test_steps,1)) * level

        (spikes,trace,string) = IAF(current,threshold)
        predicted_spikes = st.compute(trace)
        
        results[i]['trace'] = trace
        results[i]['actual'] = spikes
        results[i]['predicted'] = predicted_spikes

        print "Ratio for " +str(level) +  " = " + str( np.sum(predicted_spikes) / np.sum(spikes[100:]))
        plt.plot(spikes[100:]); plt.plot(predicted_spikes); plt.show()



def exp_variable_input(nb_epoch=7,num_steps=10000,test_steps=1000,exp_name='exp_variable_input',load_model=True,load_weights=True):

    #Training
    current = np.ones((num_steps,1)) * 0.1
    current[2500:5000] = 0.2
    current[5000:7500] = 0.3
    current[7500:] = 0.4

    threshold = 1

    (spikes,trace,string) = IAF(current,threshold)
    st = spike_trainer(trace,spikes,nb_epoch=nb_epoch,exp_name = exp_name,load_model=load_model,load_weights = load_weights)  
    st.fit_model()

    test_levels =  [0.1,0.2,0.3,0.4]
    results = [{} for l in test_levels]
    for i, level in enumerate(test_levels):
        # Prediction
        print "Predict at " + str(level)
        current = np.ones((test_steps,1)) * level

        (spikes,trace,string) = IAF(current,threshold)
        predicted_spikes = st.compute(trace)
        
        results[i]['trace'] = trace
        results[i]['actual'] = spikes
        results[i]['predicted'] = predicted_spikes

        print "Ratio for " +str(level) + " = " + str( np.sum(predicted_spikes) / np.sum(spikes[100:]))

        plt.plot(spikes[100:]); plt.plot(predicted_spikes); plt.show()

    current = np.ones((num_steps,1)) * 0.1
    current[250:] = 0.3

    (spikes,trace,string) = IAF(current,threshold)
    predicted_spikes = st.compute(trace)

    print "Ratio for " +str(level) + " = " + str( np.sum(predicted_spikes) / np.sum(spikes[100:]))
    plt.plot(spikes[100:]); plt.plot(predicted_spikes); plt.show()


def exp_paired_spikes(nb_epoch=7,num_steps=10000,test_steps=1000,exp_name='exp_paired_spikes_input',load_model=True,load_weights=True):
    #Training

    def gen_input(num_steps):
        in_spikes = np.zeros((num_steps,1))
        out_spikes = np.zeros((num_steps,1))
        i = 10
        while i < num_steps:
            i+i+1
            if random.random() < 0.05:
                in_spikes[i-1] = 1
                offset = 2+int(random.random()*10)
                in_spikes[i-offset] = 1
                out_spikes[i] = 1
                i = i +20 + int(random.random()*50)
        return (in_spikes,out_spikes)

    in_spikes,out_spikes = gen_input(num_steps)
    in_spikes_test,out_spikes_test = gen_input(test_steps)

    st = spike_trainer(in_spikes,out_spikes,exp_name=exp_name,nb_epoch=nb_epoch,load_model=load_model,load_weights=load_weights)  
    st.fit_model()
    
    predicted_spikes = st.compute(in_spikes_test)

    print "Ratio = " + str( np.sum(predicted_spikes) / np.sum(out_spikes_test[100:]))
    #plt.plot(out_spikes_test[100:]); plt.plot(-in_spikes_test[100:]); plt.show()
    plt.plot(out_spikes_test[100:]); plt.plot(-np.array(predicted_spikes)); plt.show()

def  exp_spiking_IAF_input(nb_epoch,num_steps=10000,test_steps=1000,exp_name='exp_spiking_IAF_input',load_model=True,load_weights=True):
    
    in_spikes = np.zeros((num_steps,1))
    out_spikes = np.zeros((num_steps,1))

    def gen_spikes(num_steps):

        in_spikes = np.zeros((num_steps,1))
        out_spikes = np.zeros((num_steps,1))

        s = 0
        s_t = 2
        isi = 0
        for i in np.arange(num_steps-1):
            if random.random() <0.1:
                in_spikes[i] = 1
                s = s+1
            if s == s_t:
                out_spikes[i+1] = 1
                s = 0
        in_spikes_filter = np.zeros_like(in_spikes)
       
        def filter(a): 
            f = np.zeros_like(a)
            for i in np.arange(1,num_steps):
                f[i] = 0.80*f[i-1] + a[i]
            return f/np.max(f)

            

        return (filter(in_spikes), filter(out_spikes))

    in_spikes,out_spikes = gen_spikes(num_steps)
    in_spikes_test,out_spikes_test = gen_spikes(test_steps)

    st = spike_trainer(in_spikes,out_spikes,exp_name=exp_name,nb_epoch = nb_epoch,load_weights=load_weights,load_model=load_model)  
    st.fit_model()
    predicted_spikes = st.compute(in_spikes_test)
    print "reproduced " + str(sum(predicted_spikes)) + " of " + str(sum(out_spikes_test[100:]))
    
    plt.plot(out_spikes_test[100:]); plt.plot(predicted_spikes); plt.show()

def  exp_spiking_eIAF_input(nb_epoch,num_steps=10000,test_steps=1000,exp_name='exp_spiking_eIAF_input',load_model=True,load_weights=True):
    """ using a basic sypnape integrate and fire model """     
    from neuron import eIAF
    
    in_spikes = 1.0* (np.random.rand(num_steps,1) < 0.1)
    in_spikes_test = 1.0* (np.random.rand(test_steps,1) < 0.1)

    tau = 0.8
    threshold = 1

    out_spikes,out_filter,in_filter = eIAF(in_spikes,1,tau)
    out_spikes_test,out_filter_test,in_filter_test = eIAF(in_spikes_test,1,tau)

    st = spike_trainer(in_filter,out_filter,exp_name=exp_name,nb_epoch = nb_epoch,load_weights=load_weights,load_model=load_model)  
    st.fit_model()
    predicted_spikes = st.compute(in_filter_test)
    print "reproduced " + str(sum(predicted_spikes)) + " of " + str(sum(out_filter_test[100:]))
    
    plt.plot(out_spikes_test[100:]); plt.plot(predicted_spikes); plt.show()
    plt.plot(out_filter_test[100:]); plt.plot(predicted_spikes); plt.show()
    ipdb.set_trace()

    

if __name__ == "__main__":
    #exp_constant_input(nb_epoch = 10,num_steps = 10000,test_steps=500, exp_name = 'exp_constant_input',load_model=False,load_weights =False)
    #exp_variable_input(nb_epoch = 3,num_steps = 10000, exp_name = 'exp_variable_input',load_model = True,load_weights=True)   
    #poisson_input(nb_epoch = 3,num_steps = 10000, exp_name = 'exp_poisson_input')   
    #exp_paired_spikes(nb_epoch = 7,num_steps = 10000, exp_name = 'exp_paired_spikes_input',load_model=True,load_weights=True)   
    #exp_spiking_IAF_input(nb_epoch = 5,num_steps = 10000, exp_name = 'exp_spiking_IAF_input_mse_relu_full',load_model=True,load_weights=True)
    exp_spiking_eIAF_input(nb_epoch = 1,num_steps = 5000,test_steps=2000, exp_name = 'exp_spiking_eIAF_inputi_filter', load_model=True, load_weights=True)




