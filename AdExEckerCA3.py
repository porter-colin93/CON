import matplotlib.pyplot as plt
import numpy as np

from brian2 import NeuronGroup, StateMonitor, SpikeMonitor, TimedArray, run, pF, nS, mV, ms, nA, pA, defaultclock
from brian2tools import *

"""
This model follows figure 7a of https://elifesciences.org/articles/71850#s4
"""


# neuron parameters
C = 180.13*pF
gL = 4.31*nS
EL = -75.19*mV
VT = -24.42*mV
deltaT = 4.23*mV
tauw = 84.93*ms
A = -0.27*nS
b = 206.84*pA

vspike = -3.25*mV
vreset = -29*mV

# simulation parameters
N = 1

# model equations
eqs = """
    # Brette & Gerstner Model Equations
    dv/dt = (gL*(EL-v)+gL*deltaT*exp((v-VT)/deltaT) -w + I(t))/C : volt
    dw/dt=(A*(v-EL)-w)/tauw : amp
    """

H = NeuronGroup(N, eqs,
                threshold='v > vspike',
                reset='v=vreset; w += b',
                refractory=5.96*ms,
                method='euler')

H.v = EL

I = TimedArray([0*nA, 0.55*nA, 0.55*nA, 0.55*nA, 0*nA], dt=200*ms)

# record voltage and spikes
Voltage_H = StateMonitor(H, variables='v', record=0)
Spikes_H = SpikeMonitor(H)

# run simulation
run(1000*ms)

fig, ax = plt.subplots()

# plot spike times and membrane potential
ax.plot(Spikes_H.t/ms, Spikes_H.i, color="C0", marker="x", linestyle="none", label="Refractory")
ax.plot(Voltage_H.t/ms, Voltage_H.v.T/mV, color="C0", linestyle="--", label="Refractory")
ax.set_xlabel("Time")
ax.set_ylabel("Membrane Potential")
ax.set_ylim([-90, 1]) # set upper limit to 1 in order to see spike locations

# plot step current
StepCurrentX = np.linspace(0, 1000, int(1000*ms/defaultclock.dt + 1))
StepCurrentY = 0.55*np.heaviside(StepCurrentX-200, 0.5) - 0.55*np.heaviside(StepCurrentX-800, 0.5)
ax2 = ax.twinx()
ax2.plot(StepCurrentX, StepCurrentY, label="Current", color="red")
ax2.set_ylabel("Current", color="red")

plt.show()
fig.savefig("CON/AdEx_Figures/EckerCA3VoltageTrace.pdf")
plt.close()
