from brian import *
from brian.library.IF import *
from brian.library.synapses import alpha_synapse

from numpy import *
import ipdb

def LIF(time = 1): 
    reinit_default_clock(t=0.0 * second)
    N_pre = 1 # number of presynaptic neurons
    N_post = 1 # number of postsynaptic neurons

    tau_m = 10 * ms # membrane time constant
    v_r = 0 * mV # reset potential
    v_th = 1 * mV # threshold potential
    W = 1.62 * mV # + 0.5*randn(N_pre,N_post) * mV # synaptic efficacies

    # Leaky IF neuron with alpha synapses
    eqs = leaky_IF(tau=tau_m,El=v_r)+Current("I=ge:mV")+\
    alpha_synapse(input="I_in",tau=10*ms,unit=mV,output="ge")

    lif_pre = PoissonGroup(N_pre, rates=100*Hz)
    lif_post = NeuronGroup(N_post, model=eqs, threshold=v_th, reset=v_r)
    C = Connection(lif_pre, lif_post, "ge", weight=W)

    spikes_pre = SpikeMonitor(lif_pre)
    spikes_post = SpikeMonitor(lif_post)
    v_trace = StateMonitor(lif_post, "vm", record=True)
    I_trace = StateMonitor(lif_post, "ge", record=True)

    run(time*second)

    figure(1)
    plot(v_trace.times/ms,v_trace[0]/mV)
    xlabel("$\mathrm{Time \; (ms)}$",fontsize=30)
    ylabel("$\mathrm{v \; (mV)}$",fontsize=30)

    figure(2)
    plot(I_trace.times/ms,I_trace[0]/mV)
    xlabel("$\mathrm{Time \; (ms)}$",fontsize=30)
    ylabel("$\mathrm{I_{post}(t)}$",fontsize=30)

    figure(3)
    raster_plot(spikes_pre)
    xlabel("$\mathrm{Time \; (ms)}$",fontsize=30)
    ylabel("$\mathrm{Neuron \; no}$",fontsize=30)
    show()

    num_steps = I_trace.getvalues().shape[1]
    print num_steps

    spikes_pre_out  = np.zeros((num_steps*10,1))
    spikes_post_out = np.zeros((num_steps*10,1))

    s_pre = [int(x*num_steps) for x in spikes_pre.getspiketimes()[0]]
    s_post = [int(x*num_steps) for x in spikes_post.getspiketimes()[0]]

    spikes_pre_out[s_pre] = 1
    spikes_post_out[s_post] = 1




    return (I_trace.getvalues(), v_trace.getvalues(), spikes_pre_out, spikes_post_out)

if __name__ == "__main__":
    LIF(1)
