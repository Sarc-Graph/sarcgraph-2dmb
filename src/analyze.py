import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def robust_derivative(f: np.ndarray, t: np.ndarray, window_length: int = None, polyorder: int = 3) -> np.ndarray:
    """
    Computes a robust derivative using a Savitzky-Golay filter.

    This method is more robust to noise than a simple np.gradient,
    as it fits a polynomial to a small window of the data and
    differentiates the polynomial.

    Args:
        f (np.ndarray): The signal values (y-values).
        t (np.ndarray): The corresponding time points (x-values).
        window_length (int, optional): The length of the filter window.
                                       If None, it is automatically determined.
                                       Must be an odd number.
        polyorder (int, optional): The order of the polynomial to fit.
                                   Must be less than `window_length`.

    Returns:
        np.ndarray: The derivative of the signal, `df/dt`.
    """
    # Convert to numpy arrays
    f = np.asarray(f)
    t = np.asarray(t)
    
    # Automatically determine window length if not specified
    if window_length is None:
        # Heuristic: use an odd window length, at least 5, and no more than 1/10 of data length.
        # Ensure it's not larger than the signal itself.
        window_length = min(max(5, len(f) // 10), len(f) - 1)
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
    
    # Ensure window_length is not smaller than polyorder + 1
    if window_length <= polyorder:
        raise ValueError(f"window_length ({window_length}) must be greater than polyorder ({polyorder})")
    
    # Compute derivative using Savitzky-Golay filter
    delta_t = np.mean(np.diff(t))
    df_dt = savgol_filter(f, window_length, polyorder, deriv=1, delta=delta_t)
    
    return df_dt


def _convert_to_python_primitives(data):
    """
    Recursively converts NumPy-specific types (e.g., int64) in a dictionary
    or list to standard Python types.
    """
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, dict):
        return {key: _convert_to_python_primitives(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_to_python_primitives(item) for item in data]
    else:
        return data
        

def analyze_signal(
    signal: np.ndarray,
    time_array: np.ndarray = None,
    output_path: Path = None,
    max_ratio_threshold: float = 0.8,
    derivative_threshold: float = 0.005,
    window_length: int = None
) -> dict:
    """
    Analyzes a denoised signal to identify contractions, expansions, and relaxed states.
    
    The function first mean-centers the signal, identifies zero-crossings, and then
    analyzes the blocks between these crossings to find physiological events.

    Args:
        signal (np.ndarray): The denoised signal to analyze.
        time_array (np.ndarray, optional): Time values corresponding to the signal samples.
                                           If None, a simple range is used.
        output_path (Path, optional): If provided, the analysis results will be saved
                                      to this path as a JSON file. Defaults to None.
        max_ratio_threshold (float, optional): A threshold for identifying regions
                                               around a peak (as a ratio of the max value).
        derivative_threshold (float, optional): A threshold for the derivative used to
                                                identify the boundaries of a relaxed state.
        window_length (int, optional): Window length (odd) for the Savitzky–Golay filter
                                       used in `robust_derivative`.

    Returns:
        dict: A dictionary containing the identified regions for contractions and expansions.
    """
    if time_array is None:
        time_array = np.arange(len(signal))
    
    # Use a robust derivative for a smoother result
    derivative = robust_derivative(signal, time_array, window_length=window_length, polyorder=3)
    
    # Mean-center the signal to simplify zero-crossing detection
    signal_centered = signal - np.mean(signal)

    # 1. Find zero crossings (where the centered signal changes sign)
    zero_crossings = np.where(np.diff(np.signbit(signal_centered)))[0]
    
    if len(zero_crossings) < 2:
        return {'contractions': [], 'expansions': []}

    # 2. Find blocks where the centered signal is positive
    positive_blocks = []
    # Handle the first block if it is positive
    if signal_centered[0] > 0 and zero_crossings[0] > 0:
        positive_blocks.append((0, zero_crossings[0]))
    
    for i in range(len(zero_crossings) - 1):
        start_idx = zero_crossings[i]
        end_idx = zero_crossings[i + 1]
        
        # Check if this block is positive
        if np.mean(signal_centered[start_idx:end_idx]) > 0:
            positive_blocks.append((start_idx, end_idx))
    
    # Handle the last block if it is positive
    if signal_centered[-1] > 0 and zero_crossings[-1] < len(signal) - 1:
        positive_blocks.append((zero_crossings[-1], len(signal) - 1))
    
    # 3. Analyze each positive block to find relaxed states
    relaxed_states = []
    for start_idx, end_idx in positive_blocks:
        block_signal = signal[start_idx:end_idx]
        block_derivative = derivative[start_idx:end_idx]
        
        if len(block_signal) == 0:
            continue
            
        # Find the maximum signal value and its index
        max_val = np.max(block_signal)
        max_idx_local = np.argmax(block_signal)
        max_idx_global = start_idx + max_idx_local
        
        # Find the boundaries of the "high" region (close to the peak)
        threshold_value = max_val * max_ratio_threshold
        high_indices = np.where(block_signal >= threshold_value)[0]
        
        if len(high_indices) == 0:
            continue
            
        high_start_idx = start_idx + high_indices[0]
        high_end_idx = start_idx + high_indices[-1]

        # Search for relaxed state start: from high_start_idx towards the max_idx
        relaxed_start_idx = high_start_idx
        for i in range(high_start_idx, max_idx_global):
            if np.abs(derivative[i]) < derivative_threshold:
                relaxed_start_idx = i
                break
        
        # Search for relaxed state end: from high_end_idx towards the max_idx
        relaxed_end_idx = high_end_idx
        for i in range(high_end_idx, max_idx_global, -1):
            if np.abs(derivative[i]) < derivative_threshold:
                relaxed_end_idx = i
                break

        relaxed_states.append({
            'start_idx': relaxed_start_idx,
            'end_idx': relaxed_end_idx,
            'peak_idx': max_idx_global
        })
    
    # 4. Identify contractions and expansions between relaxed states
    contractions = []
    expansions = []
    
    # We need at least two relaxed states to define a full contraction/expansion cycle
    if len(relaxed_states) > 1:
        for i in range(len(relaxed_states) - 1):
            current_relaxed_end = relaxed_states[i]['end_idx']
            next_relaxed_start = relaxed_states[i + 1]['start_idx']
            
            # Find the minimum between the relaxed states
            transition_signal = signal[current_relaxed_end:next_relaxed_start]
            
            if len(transition_signal) > 0:
                min_idx_local = np.argmin(transition_signal)
                min_idx_global = current_relaxed_end + min_idx_local
                
                # Contraction: from relaxed state end to minimum
                contractions.append({
                    'start_idx': current_relaxed_end,
                    'end_idx': min_idx_global
                })
                
                # Expansion: from minimum to next relaxed state start
                expansions.append({
                    'start_idx': min_idx_global,
                    'end_idx': next_relaxed_start
                })

    # 5. Filter out very small regions
    min_region_length = 10
    
    final_contractions = [
        c for c in contractions if c['end_idx'] - c['start_idx'] >= min_region_length
    ]
    final_expansions = [
        e for e in expansions if e['end_idx'] - e['start_idx'] >= min_region_length
    ]

    # --- Save the results if an output path is provided ---
    analysis_results = {
        'contractions': final_contractions,
        'expansions': final_expansions,
        'relaxed_states': relaxed_states
    }
    
    # Convert all NumPy types to standard Python primitives before saving
    serializable_results = _convert_to_python_primitives(analysis_results)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Saved analysis results to {output_path}")

    return analysis_results


def plot_signal_with_regions(signal, noisy_signal, analysis_results, time_array=None):
    """
    Plot the signal with highlighted regions for relaxed states, contractions, and expansions.
    
    Parameters:
    -----------
    signal : np.ndarray
        The signal to plot
    noisy_signal : np.ndarray
        The raw, noisy signal to plot for comparison
    analysis_results : dict
        Analysis results from analyze_signal function
    time_array : np.ndarray, optional
        Time values corresponding to the signal samples
    """
    if time_array is None:
        time_array = np.arange(len(signal))
    
    relaxed_states = analysis_results['relaxed_states']
    contractions = analysis_results['contractions']
    expansions = analysis_results['expansions']
    
    plt.figure(figsize=(15, 8))
    
    # Increase the global font size
    plt.rcParams.update({'font.size': 14})
    
    # Colors based on the image
    relaxed_color = '#999999'  # Light gray
    relaxed_edge_color = '#999999'  # Darker gray for borders
    contraction_color = '#FDB95F'  # Vibrant yellow
    expansion_color = '#74A9CF'  # Vibrant blue
    
    # Create hatching patterns with consistent spacing
    expansion_hatch = '\\\\'  # Baseline spacing for blue
    contraction_hatch = '//'  # Matching spacing for yellow
    
    # First plot all the background colored regions
    
    # Highlight relaxed states - with edge color for better visibility
    for state in relaxed_states:
        start_idx = state['start_idx']
        end_idx = state['end_idx']
        plt.axvspan(time_array[start_idx], time_array[end_idx], 
                    alpha=0.5, color=relaxed_color, 
                    edgecolor=relaxed_edge_color, linewidth=1.0,
                    label='_Relaxed State', zorder=1)
    
    # Highlight contractions
    for contraction in contractions:
        start_idx = contraction['start_idx']
        end_idx = contraction['end_idx']
        plt.axvspan(time_array[start_idx], time_array[end_idx], 
                    alpha=0.5, color=contraction_color, hatch=contraction_hatch, 
                    label='_Contraction', zorder=2)
    
    # Highlight expansions
    for expansion in expansions:
        start_idx = expansion['start_idx']
        end_idx = expansion['end_idx']
        plt.axvspan(time_array[start_idx], time_array[end_idx], 
                    alpha=0.5, color=expansion_color, hatch=expansion_hatch, 
                    label='_Expansion', zorder=3)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7, zorder=4)
    
    # Plot the noisy signal as dark gray dots BETWEEN the colored regions and the denoised signal
    plt.scatter(time_array, noisy_signal, marker='x', s=5, c='#444444', label='Noisy Signal', alpha=0.7, zorder=5)
    
    # Plot the denoised signal on top of everything
    plt.plot(time_array, signal, 'k-', linewidth=2.0, label='Denoised Signal', zorder=6)
    
    # Add legend with unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Create custom patch for relaxed state with edge color
    relaxed_patch = plt.Rectangle((0,0), 1, 1, color=relaxed_color, alpha=0.7, 
                                 edgecolor=relaxed_edge_color, linewidth=1.0)
    
    by_label['Relaxed State'] = relaxed_patch
    by_label['Contraction'] = plt.Rectangle((0,0), 1, 1, color=contraction_color, 
                                           alpha=0.7, hatch=contraction_hatch)
    by_label['Expansion'] = plt.Rectangle((0,0), 1, 1, color=expansion_color, 
                                         alpha=0.7, hatch=expansion_hatch)
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.title('Signal Analysis: Relaxed States, Contractions, and Expansions', fontsize=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Signal Amplitude', fontsize=16)
    
    # Optionally, set larger tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    
    return plt.gcf()