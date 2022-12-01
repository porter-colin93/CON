import matplotlib.pyplot as plt
import numpy as np

from brian2 import NeuronGroup, StateMonitor, SpikeMonitor, TimedArray, run, pF, nS, mV, ms, nA, us, defaultclock
from brian2tools import *

# neuron parameters
C = 281*pF
gL = 30*nS
EL = -70.6*mV
VT = -50.4*mV
deltaT = 2*mV
tauw = 144*ms
A = 4*nS
b = 0.0805*nA

vspike = 30*mV
vreset = -70*mV

defaultclock.dt = 0.1*ms

# simulation parameters
N = 1

# model equations
eqs = """
    # Brette & Gerstner Model Equations
    dv/dt = (gL*(EL-v)+gL*deltaT*exp((v-VT)/deltaT) -w + I(t))/C : volt
    dw/dt=(A*(v-EL)-w)/tauw : amp
    """

G = NeuronGroup(N, eqs,
                threshold='v > vspike',
                reset='v=vreset; w += b',
                refractory=0*ms,
                method='euler')

H = NeuronGroup(N, eqs,
                threshold='v > vspike',
                reset='v=vreset; w += b',
                refractory=5*ms,
                method='euler')

G.v = EL
H.v = EL

I = TimedArray([0*nA, 0.2*nA, 0*nA, 0.75*nA, 0*nA], dt=200*ms)

# record voltage and spikes
Voltage_G = StateMonitor(G, variables='v', record=0)
Spikes_G = SpikeMonitor(G)

Voltage_H = StateMonitor(H, variables='v', record=0)
Spikes_H = SpikeMonitor(H)

# run simulation
run(1000*ms)

fig, ax = plt.subplots()

# plot spike times and membrane potential
ax.plot(Spikes_G.t/ms, Spikes_G.i, color="C1", marker="*", linestyle="none", label="Not Refractory")
ax.plot(Spikes_H.t/ms, Spikes_H.i, color="C0", marker="x", linestyle="none", label="Refractory")
ax.plot(Voltage_G.t/ms, Voltage_G.v.T/mV, color="C1", label="Not Refractory")
ax.plot(Voltage_H.t/ms, Voltage_H.v.T/mV, color="C0", linestyle="--", label="Refractory")
ax.set_xlabel("Time")
ax.set_ylabel("Membrane Potential")
ax.set_ylim([-90, 1]) # set upper limit to 1 in order to see spike locations
plt.legend(loc='upper left')

# plot step current
StepCurrentX = np.linspace(0, 1000, int(1000*ms/defaultclock.dt + 1))
StepCurrentY = 0.2*np.heaviside(StepCurrentX-200, 0.5) - 0.2*np.heaviside(StepCurrentX-400, 0.5) + \
            0.75*np.heaviside(StepCurrentX-600, 0.5) - 0.75*np.heaviside(StepCurrentX-800, 0.5)
ax2 = ax.twinx()
ax2.plot(StepCurrentX, StepCurrentY, label="Current", color="red")
ax2.set_ylabel("Current", color="red")

plt.show()
fig.savefig("CON/AdEx_Figures/BretteGerstnerVoltageTrace.pdf")
plt.close()

