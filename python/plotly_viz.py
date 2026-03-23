import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from signal_gen import generate_signal, add_noise, compute_metrics
from filters import moving_average, gaussian_smooth, median_filter, savitzky_golay

def generate_interactive_plot():
    """Generates an interactive Plotly HTML dashboard for signal analysis."""
    
    # Ensure the output directory exists
    os.makedirs('analysis_output', exist_ok=True)
    
    # 1. Generate base signals
    N = 256
    sample_rate = 256
    t, clean_sig = generate_signal('sine', N=N, freq=3, sample_rate=sample_rate)
    
    # Add Gaussian noise with specified level
    noisy_sig = add_noise(clean_sig, noise_type='gaussian', level=0.5, seed=42)
    
    # 2. Apply all 4 filters (even though not all are plotted, fulfilling requirement)
    ma_sig = moving_average(noisy_sig, window=9)
    gs_sig = gaussian_smooth(noisy_sig, window=9)
    mf_sig = median_filter(noisy_sig, window=9)
    savgol_sig = savitzky_golay(noisy_sig, window=9, poly_order=2)
    
    # Calculate performance metrics for the annotation
    metrics = compute_metrics(clean_sig, noisy_sig, savgol_sig)
    rmse_before = metrics['rmse_before']
    rmse_after = metrics['rmse_after']
    
    # 3. Setup Plotly subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Domain', 'Frequency Spectrum'),
        vertical_spacing=0.15
    )
    
    # =========================================================================
    # Row 1 — Time Domain
    # =========================================================================
    fig.add_trace(
        go.Scatter(x=t, y=noisy_sig, mode='lines', line=dict(color='orange'), opacity=0.6, name='Raw Noisy'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=ma_sig, mode='lines', line=dict(color='blue'), name='Moving Average'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=savgol_sig, mode='lines', line=dict(color='green', width=2), name='Savitzky-Golay'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=clean_sig, mode='lines', line=dict(color='cyan', dash='dash'), name='Ideal'),
        row=1, col=1
    )
    
    # =========================================================================
    # Row 2 — Frequency Spectrum
    # =========================================================================
    freqs = np.fft.fftfreq(N, d=1/sample_rate)
    
    # Extract only positive frequencies
    pos_mask = freqs >= 0
    freq_bins = freqs[pos_mask]
    
    noisy_fft_mag = np.abs(np.fft.fft(noisy_sig))[pos_mask]
    savgol_fft_mag = np.abs(np.fft.fft(savgol_sig))[pos_mask]
    
    # Plot FFT magnitudes side-by-side using Bar charts
    fig.add_trace(
        go.Bar(x=freq_bins, y=noisy_fft_mag, marker_color='orange', opacity=0.8, name='Noisy FFT'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=freq_bins, y=savgol_fft_mag, marker_color='green', opacity=0.8, name='Savgol FFT'),
        row=2, col=1
    )
    
    # =========================================================================
    # Styling and Layout
    # =========================================================================
    fig.update_layout(
        title='Signal Noise Reduction — Interactive Analysis',
        template='plotly_dark',
        height=800,
        barmode='group',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add axis labels
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    
    fig.update_xaxes(title_text='Frequency Bin', row=2, col=1)
    fig.update_yaxes(title_text='Magnitude', row=2, col=1)
    
    # Add metrics annotation to the plot
    annotation_text = f"<b>Performance Metrics</b><br>RMSE Before: {rmse_before:.4f}<br>RMSE After: {rmse_after:.4f}"
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.01, y=0.98,
        showarrow=False,
        align="left",
        font=dict(color="white", size=13),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=6
    )
    
    # 4. Save to HTML
    output_path = 'analysis_output/interactive_plot.html'
    fig.write_html(output_path)
    print(f"Interactive dashboard successfully generated and saved to {output_path}")

if __name__ == '__main__':
    generate_interactive_plot()
