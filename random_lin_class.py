"""
Function for random linear classification
Parameter:
- Dn: training dataset
- k: iteration integer
- d: dimensions of dataset

Output:
- th : theta vector in d dimensions
- th0 : scalar parameter
"""

import numpy as np

def rand_lin_class(Dn,k,d):
    thetas = {} # empty dictionary to store theta vectors and values

    for i in range(0,k): # loop over iterations
        th = np.random.uniform(-10,10,(d,)) # generate random dx1 vector
        th0 = np.random.uniform(-10,10) # generate random scalar

        thetas[i] = (th,th0) # store in dictionary as tuple

    # calculate training error for each iteration
    errors = [ err_calc(Dn,thetas[i][0],thetas[i][1]) for i in range(0,k)]

    it = np.where(errors == np.min(errors))[0] # find minimum error index

    return thetas[it[0]],np.min(errors) # currently just take first element

"""
Function to calculate the error for classifier dataset Hn and training set Dn
Both are parameters which are lists of d-dimension nested arrays

It is assumed that the dataset Hn is generated using the input values from Dn.
Such that Hn[i,-1] = Dn[i,-1], only the last element will be compared
"""

def err_calc(Dn,th,th0):
    n = len(Dn) # number of datapoints
    H_vals = np.zeros(n) # empty list for binary classifier values

    for i in range(0,n):
        H_vals[i] = np.sign( np.dot( np.transpose(th),Dn[i,:-1] ) + th0) # calculate classifier

        # Get binary classification
        if H_vals[i] == 0:
            H_vals[i] = -1
    
    err1 = sum([ 1 if H_vals[i] != Dn[i,-1] else 0 for i in range(0,n) ]) # find diff between dataset

    return (1/n) * err1