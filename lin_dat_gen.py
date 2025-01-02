"""
This script will randomly assign a theta vector in d-dimensions and a theta_0 value.

Using these, it will generate a training dataset for random points.

Outputs the training dataset and theta vector and scalar
"""

import numpy as np

def gen_data(d,N):
    th = np.random.uniform(-10,10,(d,)) # generate random dx1 vector   
    th0 = np.random.uniform(-10,10) # generate random scalar

    Xn = [ np.random.uniform(-10,10,(d,)) for i in range(0,N) ]
    Dn = np.zeros([N,d+1])

    for i in range(0,N):
        val = np.sign( np.dot( np.transpose(th),Xn[i] ) + th0) # calculate classifier

        # Get binary classification
        if val == 0:
           val = -1

        Dn[i] = np.append(Xn[i],val)

    return Dn,th,th0
