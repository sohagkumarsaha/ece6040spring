# Name: Sohag Kumar Saha
# Assignment-2/HW-2 (Problem 2) 
# ECE 6040 (Signal Analysis), Spring 2023
# Tennessee Tech University 

#import numpy, matplotlib library


# Name: Sohag Kumar Saha
# Homework 2, Problem 2

import numpy as np

def inner_product(N):
    # Create the basis set b_k [n]
    n = np.arange(N)
    b_k = np.zeros((N, N))
    for k in range(N):
        if k == 0:
            b_k[k, :] = 1 / np.sqrt(N)
        else:
            b_k[k, :] = np.sqrt(2 / N) * np.cos(np.pi / N * (n + 0.5) * k)

            
    for l in range(N):   
        if l == 0:
            b_k[l, :] = np.conj (1 / np.sqrt(N))
        else:
            b_k[l, :] = np.conj(np.sqrt(2 / N) * np.cos(np.pi / N * (n + 0.5) * l))
    
    # Calculate the inner product between each element of the basis set
    inner_products = np.zeros((N, N))
    for k in range(N):
        for l in range(N):
            inner_products[k, l] = np.inner(b_k[k, :], b_k[l, :])
    
    return inner_products


def main():

    # Calculate the inner product for N = 16
    N = 16
    inner_products = inner_product(N)
    print("N =", N)
    for k in range(N):
        for l in range(N):
            print("k =", k, "l =", l, "inner_product =", inner_products[k, l])

    # Calculate the inner product for N = 64
    N = 64
    inner_products = inner_product(N)
    print("N =", N)
    for k in range(N):
        for l in range(N):
            print("k =", k, "l =", l, "inner_product =", inner_products[k, l])

    # Calculate the inner product for N = 128
    N = 128
    inner_products = inner_product(N)
    print("N =", N)
    for k in range(N):
        for l in range(N):
            print("k =", k, "l =", l, "inner_product =", inner_products[k, l])
        
        
if __name__ == '__main__':
    main()       
