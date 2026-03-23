import numpy as np

def generate_signal(signal_type, N=256, freq=3, sample_rate=256):
    """
    Generate a clean signal.
    """
    t = np.linspace(0, N / sample_rate, N, endpoint=False)
    
    if signal_type == 'sine':
        signal = np.sin(2 * np.pi * freq * t)
    elif signal_type == 'square':
        signal = np.sign(np.sin(2 * np.pi * freq * t))
    elif signal_type == 'triangle':
        signal = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif signal_type == 'ecg':
        # Simulated rough ECG pattern
        phase = (t * freq) % 1.0
        signal = np.zeros(N)
        for i, p in enumerate(phase):
            if 0.1 <= p < 0.15:
                signal[i] = 0.25 * np.sin(np.pi * (p - 0.1) / 0.05)
            elif 0.3 <= p < 0.35:
                signal[i] = -0.2 * np.sin(np.pi * (p - 0.3) / 0.05)
            elif 0.35 <= p <= 0.4:
                signal[i] = 1.0 * np.sin(np.pi * (p - 0.35) / 0.05)
            elif 0.4 < p <= 0.45:
                signal[i] = -0.3 * np.sin(np.pi * (p - 0.4) / 0.05)
            elif 0.6 <= p < 0.7:
                signal[i] = 0.35 * np.sin(np.pi * (p - 0.6) / 0.1)
    else:
        raise ValueError("signal_type options: 'sine', 'square', 'triangle', 'ecg'")
        
    return t, signal

def add_noise(signal, noise_type='gaussian', level=0.4, seed=42):
    """
    Add specified noise to the signal.
    """
    np.random.seed(seed)
    noisy_signal = signal.copy()
    
    if noise_type == 'gaussian':
        noisy_signal += np.random.normal(0, level, len(signal))
    elif noise_type == 'uniform':
        noisy_signal += np.random.uniform(-level, level, len(signal))
    elif noise_type == 'impulse':
        num_impulses = int(0.05 * len(signal))
        indices = np.random.choice(len(signal), num_impulses, replace=False)
        signs = np.random.choice([-1.0, 1.0], num_impulses)
        noisy_signal[indices] = signs * 2.0
    else:
        raise ValueError("noise_type options: 'gaussian', 'uniform', 'impulse'")
        
    return noisy_signal

def compute_metrics(clean, noisy, processed):
    """
    Compute RMSE and SNR metrics comparing clean, noisy, and processed signals.
    """
    rmse_before = np.sqrt(np.mean((clean - noisy)**2))
    rmse_after = np.sqrt(np.mean((clean - processed)**2))
    
    noise_reduced_pct = ((rmse_before - rmse_after) / rmse_before) * 100 if rmse_before > 0 else 0.0
    
    def rms(x):
        return np.sqrt(np.mean(x**2))
        
    clean_rms = rms(clean)
    rms_diff_before = rms(clean - noisy)
    rms_diff_after = rms(clean - processed)
    
    eps = 1e-10
    snr_before_db = 20 * np.log10(max(clean_rms / max(rms_diff_before, eps), eps))
    snr_after_db = 20 * np.log10(max(clean_rms / max(rms_diff_after, eps), eps))
    
    return {
        'rmse_before': rmse_before,
        'rmse_after': rmse_after,
        'noise_reduced_pct': noise_reduced_pct,
        'snr_before_db': snr_before_db,
        'snr_after_db': snr_after_db
    }

if __name__ == '__main__':
    # Sample execution block
    t, clean_sig = generate_signal('sine', N=256, freq=3, sample_rate=256)
    noisy_sig = add_noise(clean_sig, noise_type='gaussian', level=0.4, seed=42)
    
    # Creating a dummy "processed" signal: a smooth moving average
    window_len = 5
    window = np.ones(window_len) / window_len
    processed_sig = np.convolve(noisy_sig, window, mode='same')
    
    metrics = compute_metrics(clean_sig, noisy_sig, processed_sig)
    
    print("Sample Signal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
