import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

file_path = 'eeg-data.txt'
eeg_data = np.loadtxt(file_path)

# Define frequency bands
delta_band = (1, 4)
theta_band = (4, 8)
alpha_band = (8, 13)
beta_band = (13, 30)

# Compute power spectral density using Welch's periodogram
frequencies, psd = welch(eeg_data, fs=100, nperseg=256)  # Adjust nperseg as needed

# Find indices corresponding to frequency bands
delta_indices = np.where((frequencies >= delta_band[0]) & (frequencies <= delta_band[1]))
theta_indices = np.where((frequencies >= theta_band[0]) & (frequencies <= theta_band[1]))
alpha_indices = np.where((frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1]))
beta_indices = np.where((frequencies >= beta_band[0]) & (frequencies <= beta_band[1]))

# Calculate absolute bandpowers
delta_power = np.trapz(psd[delta_indices])
theta_power = np.trapz(psd[theta_indices])
alpha_power = np.trapz(psd[alpha_indices])
beta_power = np.trapz(psd[beta_indices])

# Calculate total power
total_power = np.trapz(psd)

# Calculate relative bandpowers
delta_relative_power = delta_power / total_power
theta_relative_power = theta_power / total_power
alpha_relative_power = alpha_power / total_power
beta_relative_power = beta_power / total_power

# Print or use the results as needed
print("Delta Power:", delta_power)
print("Theta Power:", theta_power)
print("Alpha Power:", alpha_power)
print("Beta Power:", beta_power)

print("\nRelative Bandpowers:")
print("Delta:", delta_relative_power)
print("Theta:", theta_relative_power)
print("Alpha:", alpha_relative_power)
print("Beta:", beta_relative_power)

# Optionally, plot the power spectral density
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, psd)
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.show()