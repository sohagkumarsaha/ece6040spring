# Name: Sohag Kumar Saha
# Solution of Problem in Homework- 4
# ECE 6040 (Signal Analysis), Spring 2023
# Tennessee Tech University 


# Import Libraries (numpy and matplotlib)

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load received signal
    y = np.load('signal.npy')

    # Define rectangular pulse function
    def rect_pulse(t):
        pulse = np.zeros(len(t))
        pulse[abs(t) < 0.5] = 1
        return pulse

    # Define LFM pulse function using rectangular pulse
    def lfm_pulse(t, B, T, u):
        k = B / T            # Given sweep rate in problem statement 
        lfm_pulse_output = rect_pulse(t/T) * np.exp(1j * u * k * (t ** 2))
        return lfm_pulse_output
    
    
    
    # Define search space for pulse parameters
    B_values = [1e6, 2e6, 5e6, 8e6]
    T_values = [10e-6, 20e-6, 30e-6, 40e-6]
    u_values = [-1, 1]

    
    # Define calibration pulse parameters
    # For calibration purposes, bandwidth, duration, chirp direction, and start time initialized  
    B1 = 5e6     # B1=5 MHz
    T1 = 20e-6   # T1=20 μs
    u1 = 1
    rising_edge1 = 0  # 0 μs

    # Define CA-CFAR detector parameters
    R = 10       # The number of reference or “training cells”, R
    G = 4        # The number of guard cells, G
    P_FA = 1e-4  # The false alarm probability, P_FA

    
    # Define parameters for CA-CFAR detector
    G = 4          # number of guard cells
    min_PFA = 1e-6 # minimum false alarm probability
    max_PFA = 1e-4 # maximum false alarm probability

    
    # Define variables to store peak values and times for each pulse
    pulse_peaks = []
    pulse_times = []

    # Loop over all possible pulse parameters
    for B in B_values:
        for T in T_values:
            for u in u_values:
                # Construct matched filter for current pulse parameters
                t_filter = np.linspace(-T/2, T/2, int(T * 10e6))
                filter_t = np.conj(lfm_pulse(t_filter, B, T, u)[::-1])

                # Apply matched filter to received signal
                matched_filter_output = np.convolve(y, filter_t, mode='same')

                # Find peak value of matched filter output
                peak_val = np.max(np.abs(matched_filter_output))

                # Apply CA-CFAR detector to matched filter output
                N = len(filter_t)
                R = N - G - 1 # number of reference cells
                PFA = 1
                while PFA > min_PFA and R > 0:
                    thresh = np.sort(np.abs(matched_filter_output))[R]
                    hit_indices = np.where(np.abs(matched_filter_output) > thresh)[0]
                    false_alarm_count = len(hit_indices) - np.ceil((2*G+1)*PFA*R).astype(int)
                    PFA = false_alarm_count / R
                    R -= 1

                # Store peak value and time of pulse if it is detected
                if PFA <= max_PFA:
                    pulse_peaks.append(peak_val)
                    pulse_times.append(np.argmax(np.abs(matched_filter_output)))

    # Plot received signal
    plt.figure()
    plt.plot(np.real(y), 'b', label='Real')
    plt.plot(np.imag(y), 'g', label='Imaginary')
    plt.title('Received Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    
    # Plot matched filter output for first pulse
    plt.figure()
    plt.plot(np.abs(matched_filter_output))
    plt.title('Matched Filter Output')
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude')
    plt.grid()

    # Plot histogram of pulse peak values
    plt.figure()
    plt.hist(pulse_peaks, bins=20)
    plt.title('Pulse Peak Values')
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    plt.grid()

    # Plot histogram of pulse times
    plt.figure()
    plt.hist(pulse_times, bins=20)
    plt.title('Pulse Times')
    plt.xlabel('Sample Index')
    plt.ylabel('')
    
    plt.show()
        


if __name__ == '__main__':
    main()


    
    
    
