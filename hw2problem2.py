# Name: Sohag Kumar Saha
# Homework 2, Problem 2
# ECE 6040 (Signal Analysis), Spring 2023
# Tennessee Tech University 

#import numpy library

import numpy as np

# declaring user define function for inner product calculation 
def inner_product(N):
    # Create the basis set b_k [n]
    n = np.arange(N)
    b_k = np.zeros((N, N))
    b_l = np.zeros((N,N))
    
    #calculation of b_k[n]
    for k in range(N):
        if k == 0:
            b_k[k, :] = 1 / np.sqrt(N)
        else:
            b_k[k, :] = np.sqrt(2 / N) * np.cos(((np.pi) / N) * (n + 0.5) * k)

    # calculation of conjugate of bk[n] which is termed as bl[n]        
    for l in range(N):   
        if l == 0:
            b_l[l, :] = np.conj (1 / np.sqrt(N))
        else:
            b_l[l, :] = np.conj(np.sqrt(2 / N) * np.cos(((np.pi) / N)  * (n + 0.5) * l))
    
    # Calculate the inner product between each element of the basis set
    inner_products = np.zeros((N, N))
    for k in range(N):
        for l in range(N):
            inner_products[k, l] = (np.inner(b_k[k, :], b_l[l, :]))   
    return inner_products


def main():

    # Calculate the inner product for N = 16
    N = 16
    inner_products = inner_product(N)
    print("N =", N)
    for k in range(N):
        for l in range(N):
            print("k =", k, ", l =", l, ", inner_product = ", inner_products[k, l])

    # Calculate the inner product for N = 64
    N = 64
    inner_products = inner_product(N)
    print("N =", N)
    for k in range(N):
        for l in range(N):
            print("k =", k, ", l =", l, ", inner_product = ", inner_products[k, l])

    # Calculate the inner product for N = 128
    N = 128
    inner_products = inner_product(N)
    print("N =", N)
    for k in range(N):
        for l in range(N):
            print("k =", k, ", l =", l, ", inner_product = ", inner_products[k, l])

        
if __name__ == '__main__':
    main()       
