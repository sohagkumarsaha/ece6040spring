import numpy as np
import matplotlib.pyplot as plt

# Load the noise signal 
y = np.load('signal.npy')

# Initialize the parameters
Fs = 10e6

# Function definition of rectangual pulse
def rect(t):
    x = np.zeros(len(t))
    x[abs(t) < 0.5] = 1
    return x

# Function Definition of Matched Filter
def cacluate_MF(y, B, T, u, Fs):
    t = np.arange(-T/2, T/2,1/Fs)
    k = B/T  # Sweep rate in Hz/s
    x = rect(t/T)*np.exp(1j*u*k*t**2)
    # Construct the matched filter for the pulse
    h = np.conj(x[::-1])
    # Apply the matched filter to the noisy signal
    b = np.convolve(y, h, mode='same')
    return b


def main():
           
    # Define the time axis
    N = len(y)  # Number of samples

    for B in [5e6]:
        for T in [20e-6]:
            for u in [1]:
                k = B/T  # Sweep rate in Hz/s
                tau = 0    # Shift of pulse in seconds
               
                b = cacluate_MF(y, B, T, u, Fs)
               
                # Find the peak value at the output
                #peak_idx = np.argmax(np.abs(b))
                #peak_val = b[peak_idx]
                #peak_time = t_axis[peak_idx]

                # Plot the results
         
                refLength = 12
                guardLength = 4
                offset = 150

                cfarWin = np.ones((refLength+guardLength)*2+1)
                cfarWin[refLength+1:refLength+1+2*guardLength] = 0
                cfarWin = cfarWin/np.sum(cfarWin)
                noiseLevel = np.convolve(np.abs(y)**2,cfarWin, mode='same')
                cfarThreshold = noiseLevel + offset


                plt.figure(figsize=(15,8))
                plt.plot(np.abs(b), label='Signal')
                plt.plot(cfarThreshold, 'g--', linewidth=1)
                plt.legend(['Signal', 'CFAR Threshold'])
                plt.xlabel('n(Samples)')
                plt.ylabel('Received Power (W)')
                plt.title('CA-CFAR')
                plt.show()
   
if __name__ == '__main__':
    main()
