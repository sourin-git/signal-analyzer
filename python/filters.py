import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from signal_gen import generate_signal, add_noise

def moving_average(signal, window=9):
    """Simple moving average using convolution."""
    return np.convolve(signal, np.ones(window)/window, mode='same')

def gaussian_smooth(signal, window=9):
    """Gaussian smoothing using ndimage."""
    sigma = window / 6
    return scipy.ndimage.gaussian_filter1d(signal, sigma=sigma)

def median_filter(signal, window=9):
    """Median filter using scipy.signal.medfilt."""
    # kernel size must be odd
    if window % 2 == 0:
        window += 1
    return scipy.signal.medfilt(signal, kernel_size=window)

def savitzky_golay(signal, window=9, poly_order=2):
    """
    Savitzky-Golay filter.
    Note: This is the most accurate smoothing filter among those evaluated.
    """
    # window_length must be odd
    if window % 2 == 0:
        window += 1
    return scipy.signal.savgol_filter(signal, window_length=window, polyorder=poly_order)

def linear_interpolation(signal, num_control_points=20):
    """Linear interpolation using evenly spaced control points."""
    n = len(signal)
    indices = np.linspace(0, n - 1, num_control_points, dtype=int)
    control_points = signal[indices]
    return np.interp(np.arange(n), indices, control_points)

def cubic_spline(signal, num_control_points=20):
    """Cubic spline interpolation using evenly spaced control points."""
    n = len(signal)
    indices = np.linspace(0, n - 1, num_control_points, dtype=int)
    control_points = signal[indices]
    cs = scipy.interpolate.CubicSpline(indices, control_points)
    return cs(np.arange(n))

def compare_all_filters(clean, noisy, window=9):
    """Computes and compares RMSE across all filters and interpolation methods."""
    results = {}
    
    processed_signals = {
        'moving_average': moving_average(noisy, window=window),
        'gaussian_smooth': gaussian_smooth(noisy, window=window),
        'median_filter': median_filter(noisy, window=window),
        'savitzky_golay': savitzky_golay(noisy, window=window, poly_order=2),
        'linear_interpolation': linear_interpolation(noisy, num_control_points=20),
        'cubic_spline': cubic_spline(noisy, num_control_points=20)
    }
    
    print(f"{'Method':<25} | {'RMSE vs Clean':<15}")
    print("-" * 43)
    
    for method, processed in processed_signals.items():
        rmse = np.sqrt(np.mean((clean - processed)**2))
        results[method] = rmse
        print(f"{method:<25} | {rmse:.6f}")
        
    return results

if __name__ == '__main__':
    # Sample execution comparing all filters
    t, clean_sig = generate_signal('sine', N=256, freq=3, sample_rate=256)
    noisy_sig = add_noise(clean_sig, noise_type='gaussian', level=0.4, seed=42)
    
    print("Evaluating filters on noisy gaussian signal:")
    compare_all_filters(clean_sig, noisy_sig, window=9)
