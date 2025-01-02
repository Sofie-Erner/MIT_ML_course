"""
Script to test functions
"""

import numpy as np
import matplotlib.pyplot as plt

from lin_dat_gen import gen_data
from random_lin_class import *

# ----- Variables -----
N = 10
n = 100
d = 2
Dn,th,th0 = gen_data(d,100)

print("Theta",th)
print("Theta_0:",th0)

# ----- Manipulate dataset -----
# plus minus parts of dataset
Dp = np.array([])
Dm = np.array([])

for Di in Dn:
    if Di[-1] == 1:
        Dp = np.append(Dp,[Di[:-1]])
    else:
        Dm = np.append(Dm,[Di[:-1]])

Dp = np.reshape(Dp,(int(len(Dp)/2),2))
Dm = np.reshape(Dm,(int(len(Dm)/2),2))

# ----- Get estimated theta
#  vector & scalar -----
k_vals = [10,100,500,1000]

n_k = len(k_vals)
th_E = np.zeros([n_k,2])
th0_E = np.zeros(n_k)
err = np.zeros(n_k)

for i in range(0,n_k):
    (th_E[i], th0_E[i]), err[i] = rand_lin_class(Dn,k_vals[i],2)

# ----- Plotting -----
# --- Training Data & Classifier
fig1, ax1 = plt.subplots(1)

x_vals = np.arange(-N,N)
y_vals = [ -(1/th[1])*(th[0]*x + th0) for x in x_vals ]
txt = "Classifier: $y = " 
txt = txt + str(round(-1/th[1],2)) + "( " + str(round(th[0],2)) + " + " + str(round(th0,2)) + " )$"
txt = r"" + txt

# Classifier line & normal vector
ax1.plot(x_vals,y_vals,label=txt,color='r',alpha=0.2,ls="-")

xy0 = [(-th0*th[0])/(th[0]**2 + th[1]**2),(-th0*th[1])/(th[0]**2 + th[1]**2)]
ax1.quiver(xy0[0],xy0[1],th[0],th[1],color='g',alpha=0.2)

# Training Data
ax1.scatter([Di[0] for Di in Dn],[Di[1] for Di in Dn],label="Datapoints",color='b',alpha=0.2)
ax1.scatter([Di[0] for Di in Dp],[Di[1] for Di in Dp],marker="+",color="b")
ax1.scatter([Di[0] for Di in Dm],[Di[1] for Di in Dm],marker="_",color="b")

# Algorithm line
y_vals = [ -(1/th_E[-1][1])*(th_E[-1][0]*x + th0_E[-1]) for x in x_vals ]
txt = "Algorithm classifier: $y = "
txt = txt + str(round(-1/th_E[-1][1],2)) + "( " + str(round(th_E[-1][0],2)) + " + " + str(round(th0_E[-1],2)) + " )$"
txt = r"" + txt
ax1.plot(x_vals,y_vals,label=txt,color='r')

# Add lines at (0,0)
ax1.axvline(x=0,color='grey')
ax1.axhline(y=0,color='grey')
ax1.legend()

ax1.grid()
ax1.set_xlim(-N,N)
ax1.set_ylim(-N,N)

fig1.savefig("data_plot.jpg",dpi=1000)

# --- K Performance
fig2, ax2 = plt.subplots(1)
ax2.plot(k_vals,err)

ax2.grid()
ax2.set_xlim(0,np.max(k_vals))
ax2.set_ylim(-0.1,np.max(err)+0.1)

ax2.set_xlabel("k")
ax2.set_ylabel("Training Error")

fig2.savefig("k_perform.jpg",dpi=1000)
