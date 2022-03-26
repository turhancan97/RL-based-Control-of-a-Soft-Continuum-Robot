import numpy as np

def cross_product(u,v):
    #returns cross product of columns of u with columns of v
    res=np.array([np.multiply(u[1,:],v[2,:]) - np.multiply(v[1,:],u[2,:]),
                  np.multiply(u[2,:],v[0,:]) - np.multiply(v[2,:],u[0,:]),
                  np.multiply(u[0,:],v[1,:]) - np.multiply(v[0,:],u[1,:])]);
    return res