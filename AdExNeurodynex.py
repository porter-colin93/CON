# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import matplotlib.pyplot as plt

import brian2 as b2
from brian2 import NeuronGroup, StateMonitor, SpikeMonitor, TimedArray, run, nS, mV, ms, nA, pA, Mohm
from brian2tools import *

# neuron parameters
tau_m = 5*ms # C = tau_m/R
R = 500*Mohm
v_rest = -70*mV
v_rheobase = -70*mV # VT in AdExBretteGerstner.py
delta_T = 2*mV
tau_w = 100*ms
a = 0.5*nS
b = 7*pA

v_spike = -30*mV
v_reset = -70*mV

# simulation parameters
N = 1
I = TimedArray([0*nA, 0.05*nA, 0*nA, 0.1*nA, 0*nA], dt=200*ms)

# model equations
eqs = """
    # Neurodynex Model Equations
    dv/dt = (-(v-v_rest) + delta_T*exp((v-v_rheobase)/delta_T)+ R * I(t) - R * w)/(tau_m) : volt
    dw/dt=(a*(v-v_rest)-w)/tau_w : amp
    """

G = NeuronGroup(N, eqs,
                threshold='v > v_spike',
                reset='v=v_reset; w += b',
                refractory=0*ms,
                method='euler')

# initial values of v and w is set here:
G.v = v_rest
G.w = 0.0 * b2.pA

# record voltage and spikes
Voltage_ND = StateMonitor(G, variables='v', record=0)
Spikes_ND = SpikeMonitor(G)

# run simulation
run(1000*ms)

# plot spike times and membrane potential
brian_plot(Spikes_ND)
brian_plot(Voltage_ND)
plt.savefig("CON/AdEx_Figures/NeurodynexVoltageTrace.pdf")
plt.close()
