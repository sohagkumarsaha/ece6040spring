# Name: Sohag Kumar Saha
# Assignment-2 (Problem 3) 
# ECE 6040 (Signal Analysis), Spring 2023
# Tennessee Tech University 

#import numpy, matplotlib library


# Name: Sohag Kumar Saha
# Homework 2, Problem 3

import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
x = 0.1 * n

def DFT(x):
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)
    return X

def DCT(x):
    X = np.zeros(N)
    for k in range(N):
        if k == 0:
            c = 1 / np.sqrt(N)
        else:
            c = np.sqrt(2 / N)
        for n in range(N):
            X[k] += c * x[n] * np.cos(np.pi * (n + 0.5) * k / N)
    return X

def main(): 

    
    X_f = DFT(x)
    X_c = DCT(x)

    plt.figure()
    plt.stem(np.abs(X_f))
    plt.title("DFT Magnitude")

    plt.figure()
    plt.stem(np.abs(X_c))
    plt.title("DCT Magnitude")

    plt.show()
    
    
if __name__ == '__main__':
    main()
    
