import os
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from signal_gen import generate_signal, add_noise
from filters import moving_average, gaussian_smooth, median_filter, savitzky_golay

def generate_all_plots():
    """Generates and saves visual plots comparing clean, noisy, and filtered signals."""
    
    # Ensure the output directory exists
    os.makedirs('analysis_output', exist_ok=True)
    
    # General styling setup
    plt.style.use('dark_background')
    
    # 1. Prepare standard sample data
    sample_rate = 256
    t, clean_sig = generate_signal('sine', N=256, freq=3, sample_rate=sample_rate)
    noisy_sig = add_noise(clean_sig, noise_type='gaussian', level=0.4, seed=42)
    savgol_sig = savitzky_golay(noisy_sig, window=9, poly_order=2)
    
    # =========================================================================
    # Plot 1 — Raw vs Filtered
    # =========================================================================
    plt.figure(figsize=(12, 5))
    
    plt.plot(clean_sig, color='cyan', linestyle='--', linewidth=1.5, label='Ideal')
    plt.plot(noisy_sig, color='orange', linewidth=0.8, alpha=0.7, label='Noisy')
    plt.plot(savgol_sig, color='green', linewidth=2, label='Savitzky-Golay Filtered')
    
    plt.title('Raw vs Filtered Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analysis_output/raw_vs_filtered.png', dpi=150)
    plt.close()

    # =========================================================================
    # Plot 2 — Comparison of all filters
    # =========================================================================
    filters_to_compare = {
        'Moving Average': moving_average(noisy_sig, window=9),
        'Gaussian': gaussian_smooth(noisy_sig, window=9),
        'Median': median_filter(noisy_sig, window=9),
        'Savitzky-Golay': savgol_sig
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for ax, (title, filtered_sig) in zip(axes, filters_to_compare.items()):
        # Faint orange for noisy signal background
        ax.plot(noisy_sig, color='orange', linewidth=0.8, alpha=0.3, label='Noisy')
        # Solid green for the filtered line
        ax.plot(filtered_sig, color='green', linewidth=1.5, label='Filtered')
        ax.set_title(title)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        
    plt.tight_layout()
    plt.savefig('analysis_output/filter_comparison.png')
    plt.close()

    # =========================================================================
    # Plot 3 — Frequency Spectrum
    # =========================================================================
    N = len(noisy_sig)
    
    # Calculate FFT frequencies and extract magnitude
    freqs = np.fft.fftfreq(N, d=1/sample_rate)
    noisy_fft_mag = np.abs(np.fft.fft(noisy_sig))
    filtered_fft_mag = np.abs(np.fft.fft(savgol_sig))
    
    # Mask to keep only the positive frequencies
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_noisy_fft = noisy_fft_mag[pos_mask]
    pos_filtered_fft = filtered_fft_mag[pos_mask]
    
    # Plot spectra side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(pos_freqs, pos_noisy_fft, color='orange', linewidth=1.5)
    ax1.set_title('Noisy Spectrum')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    
    ax2.plot(pos_freqs, pos_filtered_fft, color='green', linewidth=1.5)
    ax2.set_title('Filtered Spectrum (Savitzky-Golay)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    
    plt.tight_layout()
    plt.savefig('analysis_output/frequency_spectrum.png')
    plt.close()

if __name__ == '__main__':
    generate_all_plots()
    print("All signals are plotted! Saved to the 'analysis_output/' directory.")
