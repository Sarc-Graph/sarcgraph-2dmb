import numpy as np
import pandas as pd
import cv2
import json

from typing import Dict
from pathlib import Path


def extract_metrics(
    sarcomere_data_df: pd.DataFrame,
    smoothed_signal_array: np.ndarray,
    analysis_results_dict: dict,
    bounding_boxes_px: np.ndarray,
    convex_hulls_px: np.ndarray,
    PIXEL_TO_MICRON: float = 0.1625,
    FRAME_RATE: float = 1.0,
    FRAME_AVG_RANGE: int = 5
) -> dict:
    """
    Extracts a comprehensive set of physiological metrics from sarcomere length data
    and signal analysis results, converting frame-based durations to time-based using FRAME_RATE.

    Args:
        sarcomere_data_df (pd.DataFrame): DataFrame containing raw sarcomere data
                                          with 'frame' and 'length' columns.
                                          Lengths are expected to be in pixels and
                                          will be converted to microns internally.
        smoothed_signal_array (np.ndarray): The GPR-smoothed sarcomere length signal
                                            (mean length per frame), in original pixel units.
        analysis_results_dict (dict): Dictionary containing identified 'contractions',
                                      'expansions', and 'relaxed_states' from analyze_signal.
        bounding_boxes_px (np.ndarray): NumPy array of bounding box coordinates in pixel units.
        convex_hulls_px (np.ndarray): NumPy array of convex hull coordinates in pixel units.
        PIXEL_TO_MICRON (float): Conversion factor from pixels to microns.
        FRAME_RATE (float): The frame rate of the video (frames per second). Used to convert
                            frame differences to time differences (e.g., seconds).
        FRAME_AVG_RANGE (int): Number of frames around a peak/onset to average for length calculations.

    Returns:
        dict: A dictionary containing all calculated physiological metrics for the sample.
    """

    # --- 0. Initial Data Preparation and Unit Conversion ---
    # Ensure all lengths are in microns from the start of this function
    sarcomere_data_df_mu = sarcomere_data_df.copy()
    sarcomere_data_df_mu['length'] *= PIXEL_TO_MICRON
    
    gpr_signal_mu = smoothed_signal_array * PIXEL_TO_MICRON # Smoothed signal in microns

    # Extract phases from analysis_results_dict
    relaxed_states = analysis_results_dict['relaxed_states']
    contractions = analysis_results_dict['contractions']
    relaxations = analysis_results_dict['expansions'] # Your original code used 'expansions' for relaxations

    # Handle cases where no contractions/relaxations were found
    if not contractions or not relaxations or not relaxed_states:
        print("Warning: No contraction/relaxation cycles found. Returning empty metrics.")
        return {
            'num_contractions': 0,
            'contraction_period': np.nan,
            'contraction_frequency': np.nan,
            'relaxed_sarcomere_length_mean': np.nan,
            'peak_sarcomere_length_mean': np.nan,
            'shortening_amplitude': np.nan,
            'peak_shortening_velocity': np.nan,
            'peak_lengthening_velocity': np.nan,
            'contraction_onset_to_relaxation_end_time': np.nan,
            'contraction_onset_to_peak_contraction_time': np.nan,
            'contraction_onset_to_50_contracted_time': np.nan,
            'half_contracted_to_peak_contraction_time': np.nan,
            'peak_contraction_to_relaxation_end_time': np.nan,
            'peak_contraction_to_50_relaxed_time': np.nan,
            'half_relaxed_to_full_relaxation_time': np.nan,
            'tissue_length_relaxed': np.nan,
            'tissue_length_peak_contraction': np.nan,
            'tissue_width_relaxed': np.nan,
            'tissue_width_peak_contraction': np.nan,
            'tissue_area_convex_hull_relaxed': np.nan,
            'tissue_area_convex_hull_peak_contraction': np.nan,
            'tissue_area_bounding_box_relaxed': np.nan,
            'tissue_area_bounding_box_peak_contraction': np.nan,
            'total_number_of_sarcomeres': np.nan,
            'sarcomere_density_convex_hull': np.nan,
            'sarcomere_density_bounding_box': np.nan,
            'noise_level': np.nan
        }


    # Determine if the signal starts with a relaxed state (important for cycle definition)
    # This logic assumes relaxed_states and contractions are sorted by start_idx
    starts_with_relaxed = relaxed_states[0]['start_idx'] < contractions[0]['start_idx'] if relaxed_states and contractions else False

    # Determine the number of full contractions that have corresponding relaxed/expansion phases
    # This ensures we only analyze complete cycles
    num_contractions = min(len(relaxed_states), len(contractions), len(relaxations))
    if num_contractions == 0:
        print("Warning: Not enough full cycles to compute all metrics. Returning NaN for cycle-dependent metrics.")
        return {
            'num_contractions': 0,
            'contraction_period': np.nan,
            'contraction_frequency': np.nan,
            'relaxed_sarcomere_length_mean': np.nan,
            'peak_sarcomere_length_mean': np.nan,
            'shortening_amplitude': np.nan,
            'peak_shortening_velocity': np.nan,
            'peak_lengthening_velocity': np.nan,
            'contraction_onset_to_relaxation_end_time': np.nan,
            'contraction_onset_to_peak_contraction_time': np.nan,
            'contraction_onset_to_50_contracted_time': np.nan,
            'half_contracted_to_peak_contraction_time': np.nan,
            'peak_contraction_to_relaxation_end_time': np.nan,
            'peak_contraction_to_50_relaxed_time': np.nan,
            'half_relaxed_to_full_relaxation_time': np.nan,
            'tissue_length_relaxed': np.nan,
            'tissue_length_peak_contraction': np.nan,
            'tissue_width_relaxed': np.nan,
            'tissue_width_peak_contraction': np.nan,
            'tissue_area_convex_hull_relaxed': np.nan,
            'tissue_area_convex_hull_peak_contraction': np.nan,
            'tissue_area_bounding_box_relaxed': np.nan,
            'tissue_area_bounding_box_peak_contraction': np.nan,
            'total_number_of_sarcomeres': np.nan,
            'sarcomere_density_convex_hull': np.nan,
            'sarcomere_density_bounding_box': np.nan,
            'noise_level': np.nan
        }

    # --- 1. Compute Contraction Periods and Frequency ---
    contraction_periods = []
    for i in range(num_contractions):
        if starts_with_relaxed:
            # Cycle: Relaxed -> Contraction -> Relaxation
            start_time = relaxed_states[i]['start_idx']
            end_time = relaxations[i]['end_idx']
        else:
            # Cycle: Contraction -> Relaxation -> Relaxed
            start_time = contractions[i]['start_idx']
            end_time = relaxed_states[i]['end_idx']
        
        # Ensure valid period
        if end_time > start_time:
            contraction_periods.append((end_time - start_time) / FRAME_RATE) # Converted to time using FRAME_RATE

    # Calculate average contraction period and frequency
    contraction_period = np.mean(contraction_periods) if contraction_periods else np.nan
    contraction_frequency = np.mean([1 / T for T in contraction_periods]) if contraction_periods else np.nan

    # --- 2. Compute Per-Frame Statistics for Raw Sarcomere Lengths ---
    # This is used for relaxed/peak length percentiles and std
    frame_stats = sarcomere_data_df_mu.groupby('frame')['length'].agg(
        mean='mean',
        median='median',
        std='std',
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()

    # --- 3. Extract Statistics for Relaxed State ---
    relaxed_mean_sarcomere_lengths = []
    relaxed_median_sarcomere_lengths = []
    relaxed_q25_sarcomere_length = []
    relaxed_q75_sarcomere_length = []
    relaxed_std_sarcomere_length = []

    for state in relaxed_states[:num_contractions]: # Iterate over relevant relaxed states
        start_time, end_time = state['start_idx'], state['end_idx']
        mean_values = frame_stats[(frame_stats['frame'] >= start_time) & (frame_stats['frame'] < end_time)]
        
        if not mean_values.empty:
            relaxed_mean_sarcomere_lengths.extend(mean_values['mean'].values)
            relaxed_median_sarcomere_lengths.extend(mean_values['median'].values)
            relaxed_q25_sarcomere_length.extend(mean_values['q25'].values)
            relaxed_q75_sarcomere_length.extend(mean_values['q75'].values)
            relaxed_std_sarcomere_length.extend(mean_values['std'].values)

    relaxed_mean = np.mean(relaxed_mean_sarcomere_lengths) if relaxed_mean_sarcomere_lengths else np.nan
    relaxed_median = np.mean(relaxed_median_sarcomere_lengths) if relaxed_median_sarcomere_lengths else np.nan
    relaxed_q25 = np.mean(relaxed_q25_sarcomere_length) if relaxed_q25_sarcomere_length else np.nan
    relaxed_q75 = np.mean(relaxed_q75_sarcomere_length) if relaxed_q75_sarcomere_length else np.nan
    relaxed_std = np.mean(relaxed_std_sarcomere_length) if relaxed_std_sarcomere_length else np.nan

    # --- 4. Extract Statistics for Peak Contraction State ---
    peak_mean_sarcomere_lengths = []
    peak_median_sarcomere_lengths = []
    peak_q25_sarcomere_length = []
    peak_q75_sarcomere_length = []
    peak_std_sarcomere_length = []

    for contraction in contractions[:num_contractions]:
        peak_frame = contraction['end_idx']
        peak_range = frame_stats[(frame_stats['frame'] >= peak_frame - FRAME_AVG_RANGE) & (frame_stats['frame'] <= peak_frame + FRAME_AVG_RANGE)]

        if not peak_range.empty:
            peak_mean_sarcomere_lengths.extend(peak_range['mean'].values)
            peak_median_sarcomere_lengths.extend(peak_range['median'].values)
            peak_q25_sarcomere_length.extend(peak_range['q25'].values)
            peak_q75_sarcomere_length.extend(peak_range['q75'].values)
            peak_std_sarcomere_length.extend(peak_range['std'].values)

    peak_mean = np.mean(peak_mean_sarcomere_lengths) if peak_mean_sarcomere_lengths else np.nan
    peak_median = np.mean(peak_median_sarcomere_lengths) if peak_median_sarcomere_lengths else np.nan
    peak_q25 = np.mean(peak_q25_sarcomere_length) if peak_q25_sarcomere_length else np.nan
    peak_q75 = np.mean(peak_q75_sarcomere_length) if peak_q75_sarcomere_length else np.nan
    peak_std = np.mean(peak_std_sarcomere_length) if peak_std_sarcomere_length else np.nan

    # --- 5. Compute Shortening Amplitudes ---
    shortening_amplitudes = []
    for i in range(num_contractions):
        # Get relaxed mean length for the current cycle
        relaxed_cycle_mean_values = frame_stats[(frame_stats['frame'] >= relaxed_states[i]['start_idx']) & 
                                                (frame_stats['frame'] < relaxed_states[i]['end_idx'])]
        relaxed_mean_cycle = np.mean(relaxed_cycle_mean_values['mean'].values) if not relaxed_cycle_mean_values.empty else np.nan
        
        # Get peak contraction mean length for the current cycle
        peak_frame_cycle = contractions[i]['end_idx']
        peak_cycle_mean_values = frame_stats[(frame_stats['frame'] >= peak_frame_cycle - FRAME_AVG_RANGE) & 
                                              (frame_stats['frame'] <= peak_frame_cycle + FRAME_AVG_RANGE)]
        peak_mean_cycle = np.mean(peak_cycle_mean_values['mean'].values) if not peak_cycle_mean_values.empty else np.nan
        
        if not np.isnan(relaxed_mean_cycle) and not np.isnan(peak_mean_cycle):
            shortening_amplitudes.append(relaxed_mean_cycle - peak_mean_cycle)
    
    avg_shortening_amplitude = np.mean(shortening_amplitudes) if shortening_amplitudes else np.nan

    # --- 6. Compute Velocities from GPR Signal ---
    # Velocity is the derivative of the smoothed signal, converted to per unit time
    velocity_gpr = np.gradient(gpr_signal_mu) * FRAME_RATE # Multiplied by FRAME_RATE for per second velocity

    peak_contraction_velocities = []
    for i in range(num_contractions):
        start_frame, end_frame = contractions[i]['start_idx'], contractions[i]['end_idx']
        if end_frame >= start_frame and end_frame + 1 <= len(velocity_gpr): # Ensure valid slice
            contraction_velocity = velocity_gpr[start_frame:end_frame+1]
            if contraction_velocity.size > 0:
                peak_contraction_velocities.append(np.min(contraction_velocity)) # Most negative for shortening
    avg_peak_shortening_velocity = np.mean(peak_contraction_velocities) if peak_contraction_velocities else np.nan

    peak_relaxation_velocities = []
    for i in range(num_contractions):
        start_frame, end_frame = relaxations[i]['start_idx'], relaxations[i]['end_idx']
        if end_frame >= start_frame and end_frame + 1 <= len(velocity_gpr): # Ensure valid slice
            relaxation_velocity = velocity_gpr[start_frame:end_frame+1]
            if relaxation_velocity.size > 0:
                peak_relaxation_velocities.append(np.max(relaxation_velocity)) # Most positive for lengthening
    avg_peak_lengthening_velocity = np.mean(peak_relaxation_velocities) if peak_relaxation_velocities else np.nan

    # --- 7. Compute Timing Metrics ---
    onset_to_peak_times = []
    onset_to_end_times = []
    onset_to_50_contracted_times = []
    half_to_full_contract_times = []
    peak_to_end_times = []
    peak_to_50_relaxed_times = []
    half_to_full_relax_times = []

    for i in range(num_contractions):
        contraction_start_frame = contractions[i]['start_idx']
        contraction_end_frame = contractions[i]['end_idx'] # This is the peak shortening frame
        relaxation_start_frame = relaxations[i]['start_idx']
        relaxation_end_frame = relaxations[i]['end_idx']

        # Ensure valid frames for calculations
        if not (contraction_start_frame < contraction_end_frame <= relaxation_start_frame < relaxation_end_frame):
            print(f"Warning: Skipping cycle {i} due to invalid phase ordering/indices.")
            continue

        # Onset to Peak Contraction Time
        onset_to_peak_times.append((contraction_end_frame - contraction_start_frame) / FRAME_RATE)

        # Onset to End of Relaxation Time (Total Cycle Time from onset)
        onset_to_end_times.append((relaxation_end_frame - contraction_start_frame) / FRAME_RATE)

        # Onset to 50% Contracted Time
        onset_length_val = np.mean(sarcomere_data_df_mu[(sarcomere_data_df_mu['frame'] >= contraction_start_frame - FRAME_AVG_RANGE) & 
                                                        (sarcomere_data_df_mu['frame'] <= contraction_start_frame + FRAME_AVG_RANGE)]['length'].values)
        peak_length_val = np.mean(sarcomere_data_df_mu[(sarcomere_data_df_mu['frame'] >= contraction_end_frame - FRAME_AVG_RANGE) & 
                                                       (sarcomere_data_df_mu['frame'] <= contraction_end_frame + FRAME_AVG_RANGE)]['length'].values)
        
        half_contraction_length = 0.5 * (peak_length_val + onset_length_val)
        
        # Find frame where signal crosses half_contraction_length during contraction
        contraction_segment_df = sarcomere_data_df_mu[(sarcomere_data_df_mu['frame'] >= contraction_start_frame) & 
                                                      (sarcomere_data_df_mu['frame'] <= contraction_end_frame)].groupby('frame')['length'].mean().reset_index()
        
        half_contraction_frame = np.nan
        if not contraction_segment_df.empty:
            # Find the frame where length is closest to half_contraction_length
            idx_closest = (np.abs(contraction_segment_df['length'] - half_contraction_length)).argmin()
            half_contraction_frame = contraction_segment_df.iloc[idx_closest]['frame']
        
        if not np.isnan(half_contraction_frame):
            onset_to_50_contracted_times.append((half_contraction_frame - contraction_start_frame) / FRAME_RATE)
            half_to_full_contract_times.append((contraction_end_frame - half_contraction_frame) / FRAME_RATE)
        else:
            onset_to_50_contracted_times.append(np.nan)
            half_to_full_contract_times.append(np.nan)

        # Peak Contraction to End of Relaxation Time
        peak_to_end_times.append((relaxation_end_frame - contraction_end_frame) / FRAME_RATE)

        # Peak Contraction to 50% Relaxed Time
        end_length_val = np.mean(sarcomere_data_df_mu[(sarcomere_data_df_mu['frame'] >= relaxation_end_frame - FRAME_AVG_RANGE) & 
                                                      (sarcomere_data_df_mu['frame'] <= relaxation_end_frame + FRAME_AVG_RANGE)]['length'].values)
        
        half_relaxation_length = 0.5 * (peak_length_val + end_length_val) # Note: uses peak_length_val from contraction
        
        # Find frame where signal crosses half_relaxation_length during relaxation
        relaxation_segment_df = sarcomere_data_df_mu[(sarcomere_data_df_mu['frame'] >= contraction_end_frame) & 
                                                      (sarcomere_data_df_mu['frame'] <= relaxation_end_frame)].groupby('frame')['length'].mean().reset_index()
        
        half_relax_frame = np.nan
        if not relaxation_segment_df.empty:
            # Find the frame where length is closest to half_relaxation_length
            idx_closest = (np.abs(relaxation_segment_df['length'] - half_relaxation_length)).argmin()
            half_relax_frame = relaxation_segment_df.iloc[idx_closest]['frame']

        if not np.isnan(half_relax_frame):
            peak_to_50_relaxed_times.append((half_relax_frame - contraction_end_frame) / FRAME_RATE)
            half_to_full_relax_times.append((relaxation_end_frame - half_relax_frame) / FRAME_RATE)
        else:
            peak_to_50_relaxed_times.append(np.nan)
            half_to_full_relax_times.append(np.nan)

    avg_onset_to_peak_time = np.nanmean(onset_to_peak_times) if onset_to_peak_times else np.nan
    avg_onset_to_end_time = np.nanmean(onset_to_end_times) if onset_to_end_times else np.nan
    avg_onset_to_50_contracted_time = np.nanmean(onset_to_50_contracted_times) if onset_to_50_contracted_times else np.nan
    avg_half_to_full_contract_time = np.nanmean(half_to_full_contract_times) if half_to_full_contract_times else np.nan
    avg_peak_to_end_time = np.nanmean(peak_to_end_times) if peak_to_end_times else np.nan
    avg_peak_to_50_relaxed_time = np.nanmean(peak_to_50_relaxed_times) if peak_to_50_relaxed_times else np.nan
    avg_half_to_full_relax_time = np.nanmean(half_to_full_relax_times) if half_to_full_relax_times else np.nan


    # --- 8. Tissue Geometry Metrics (from bounding box data) ---
    # Convert to microns
    bbs_mu = bounding_boxes_px * PIXEL_TO_MICRON
    convex_hulls_mu = convex_hulls_px * PIXEL_TO_MICRON

    # Calculate areas (ensure input to cv2.contourArea is int32)
    # cv2.contourArea expects (N, 1, 2) or (N, 2) array of points.
    # np.int32 conversion is crucial.
    areas_bb_mu = np.array([cv2.contourArea(bb.astype(np.int32)) for bb in bbs_mu])
    areas_hull_mu = np.array([cv2.contourArea(contour.astype(np.int32)) for contour in convex_hulls_mu])

    # Per-frame tissue lengths and widths from bounding boxes
    tissue_lengths_mu = []
    tissue_widths_mu = []
    
    # Assuming bbs_mu is (num_frames, 4, 2)
    for frame_idx in range(bbs_mu.shape[0]):
        current_bb_points = bbs_mu[frame_idx]
        # Sort points by x-coordinate to consistently identify corners
        sorted_points = current_bb_points[np.argsort(current_bb_points[:, 0])]
        
        # p1, p2 are left points, p3, p4 are right points after sorting by x
        # Then sort by y for p1, p2 and p3, p4
        p_left_sorted = sorted_points[:2][np.argsort(sorted_points[:2, 1])]
        p_right_sorted = sorted_points[2:][np.argsort(sorted_points[2:, 1])]
        
        p1 = p_left_sorted[0] # top-left
        p2 = p_left_sorted[1] # bottom-left
        p3 = p_right_sorted[0] # top-right
        p4 = p_right_sorted[1] # bottom-right

        # Calculate vectors for length and width
        # The longest side of the bounding box is typically the length, shortest is width
        side1 = p2 - p1 # Vertical side on left
        side2 = p3 - p1 # Horizontal side on top
        side3 = p4 - p2 # Horizontal side on bottom
        side4 = p4 - p3 # Vertical side on right

        len1 = np.linalg.norm(side1)
        len2 = np.linalg.norm(side2)
        
        # The lengths of the sides of a rotated rectangle are len1, len2, len1, len2
        # So we just need two distinct lengths
        tissue_lengths_mu.append(max(len1, len2))
        tissue_widths_mu.append(min(len1, len2))

    tissue_lengths_mu = np.array(tissue_lengths_mu)
    tissue_widths_mu = np.array(tissue_widths_mu)
            
    # --- 9. Aggregate Tissue Geometry Metrics for Relaxed and Peak Contraction States ---
    tissue_lengths_relaxed = []
    tissue_widths_relaxed = []
    tissue_area_convex_hull_relaxed = []
    tissue_area_bound_box_relaxed = []

    for state in relaxed_states[:num_contractions]:
        start_time, end_time = state['start_idx'], state['end_idx']
        # Ensure indices are within bounds of the arrays
        start_idx_safe = max(0, start_time)
        end_idx_safe = min(len(tissue_lengths_mu), end_time)
        
        if end_idx_safe > start_idx_safe:
            tissue_lengths_relaxed.append(np.mean(tissue_lengths_mu[start_idx_safe:end_idx_safe]))
            tissue_widths_relaxed.append(np.mean(tissue_widths_mu[start_idx_safe:end_idx_safe]))
            tissue_area_convex_hull_relaxed.append(np.mean(areas_hull_mu[start_idx_safe:end_idx_safe]))
            tissue_area_bound_box_relaxed.append(np.mean(areas_bb_mu[start_idx_safe:end_idx_safe]))

    tissue_lengths_peak_contraction = []
    tissue_widths_peak_contraction = []
    tissue_area_convex_hull_peak_contraction = []
    tissue_area_bound_box_peak_contraction = []

    for contraction in contractions[:num_contractions]:
        peak_frame = contraction['end_idx']
        # Define the range around the peak for averaging
        start_avg_frame = max(0, peak_frame - FRAME_AVG_RANGE)
        end_avg_frame = min(len(tissue_lengths_mu), peak_frame + FRAME_AVG_RANGE + 1) # +1 for exclusive end
        
        if end_avg_frame > start_avg_frame:
            tissue_lengths_peak_contraction.append(np.mean(tissue_lengths_mu[start_avg_frame:end_avg_frame]))
            tissue_widths_peak_contraction.append(np.mean(tissue_widths_mu[start_avg_frame:end_avg_frame]))
            tissue_area_convex_hull_peak_contraction.append(np.mean(areas_hull_mu[start_avg_frame:end_avg_frame]))
            tissue_area_bound_box_peak_contraction.append(np.mean(areas_bb_mu[start_avg_frame:end_avg_frame]))

    avg_tissue_length_relaxed = np.mean(tissue_lengths_relaxed) if tissue_lengths_relaxed else np.nan
    avg_tissue_width_relaxed = np.mean(tissue_widths_relaxed) if tissue_widths_relaxed else np.nan
    avg_tissue_area_convex_hull_relaxed = np.mean(tissue_area_convex_hull_relaxed) if tissue_area_convex_hull_relaxed else np.nan
    avg_tissue_area_bound_box_relaxed = np.mean(tissue_area_bound_box_relaxed) if tissue_area_bound_box_relaxed else np.nan

    avg_tissue_length_peak_contraction = np.mean(tissue_lengths_peak_contraction) if tissue_lengths_peak_contraction else np.nan
    avg_tissue_width_peak_contraction = np.mean(tissue_widths_peak_contraction) if tissue_widths_peak_contraction else np.nan
    avg_tissue_area_convex_hull_peak_contraction = np.mean(tissue_area_convex_hull_peak_contraction) if tissue_area_convex_hull_peak_contraction else np.nan
    avg_tissue_area_bound_box_peak_contraction = np.mean(tissue_area_bound_box_peak_contraction) if tissue_area_bound_box_peak_contraction else np.nan

    # --- 10. Sarcomere Count and Density ---
    sarcomere_count = sarcomere_data_df_mu.groupby('frame').size().values
    
    # Ensure areas_hull_mu and areas_bb_mu are aligned with sarcomere_count frames
    # This assumes sarcomere_count has an entry for every frame where area is calculated.
    # If not, a more robust merge/alignment would be needed.
    
    # Filter areas to only include frames present in sarcomere_count
    frames_with_sarcomeres = sarcomere_data_df_mu['frame'].unique()
    areas_hull_aligned = areas_hull_mu[frames_with_sarcomeres] if frames_with_sarcomeres.max() < len(areas_hull_mu) else np.array([])
    areas_bb_aligned = areas_bb_mu[frames_with_sarcomeres] if frames_with_sarcomeres.max() < len(areas_bb_mu) else np.array([])

    avg_number_of_sarcomeres = np.mean(sarcomere_count) if sarcomere_count.size > 0 else np.nan
    
    sarcomere_density_convext_hull = np.mean(sarcomere_count / areas_hull_aligned) if areas_hull_aligned.size > 0 and (areas_hull_aligned != 0).all() else np.nan
    sarcomere_density_bound_box = np.mean(sarcomere_count / areas_bb_aligned) if areas_bb_aligned.size > 0 and (areas_bb_aligned != 0).all() else np.nan

    # --- 11. Noise Level ---
    # This requires the original mean sarcomere length per frame (in microns)
    # We need to get the raw mean length per frame in microns for this calculation.
    # Assuming `sarcomere_data_df_mu` is the source for raw mean lengths.
    raw_mean_length_mu_per_frame = sarcomere_data_df_mu.groupby('frame')['length'].mean().values
    
    # Ensure gpr_signal_mu and raw_mean_length_mu_per_frame have same length
    # This assumes they are already aligned by frame.
    noise_level = np.std(gpr_signal_mu - raw_mean_length_mu_per_frame) if gpr_signal_mu.size > 0 and raw_mean_length_mu_per_frame.size > 0 else np.nan

    # --- 12. Compile Results ---
    metrics = {
        'num_contractions': num_contractions,
        'contraction_period': contraction_period,
        'contraction_frequency': contraction_frequency,
        'relaxed_sarcomere_length_mean': relaxed_mean,
        'relaxed_sarcomere_length_median': relaxed_median,
        'relaxed_sarcomere_length_q25': relaxed_q25,
        'relaxed_sarcomere_length_q75': relaxed_q75,
        'relaxed_sarcomere_length_std': relaxed_std,
        'peak_sarcomere_length_mean': peak_mean,
        'peak_sarcomere_length_median': peak_median,
        'peak_sarcomere_length_q25': peak_q25,
        'peak_sarcomere_length_q75': peak_q75,
        'peak_sarcomere_length_std': peak_std,
        'shortening_amplitude': avg_shortening_amplitude,
        'peak_shortening_velocity': avg_peak_shortening_velocity,
        'peak_lengthening_velocity': avg_peak_lengthening_velocity,
        'contraction_onset_to_relaxation_end_time': avg_onset_to_end_time,
        'contraction_onset_to_peak_contraction_time': avg_onset_to_peak_time,
        'contraction_onset_to_50_contracted_time': avg_onset_to_50_contracted_time,
        'half_contracted_to_peak_contraction_time': avg_half_to_full_contract_time,
        'peak_contraction_to_relaxation_end_time': avg_peak_to_end_time,
        'peak_contraction_to_50_relaxed_time': avg_peak_to_50_relaxed_time,
        'half_relaxed_to_full_relaxation_time': avg_half_to_full_relax_time,
        'tissue_length_relaxed': avg_tissue_length_relaxed,
        'tissue_length_peak_contraction': avg_tissue_length_peak_contraction,
        'tissue_width_relaxed': avg_tissue_width_relaxed,
        'tissue_width_peak_contraction': avg_tissue_width_peak_contraction,
        'tissue_area_convex_hull_relaxed': avg_tissue_area_convex_hull_relaxed,
        'tissue_area_convex_hull_peak_contraction': avg_tissue_area_convex_hull_peak_contraction,
        'tissue_area_bounding_box_relaxed': avg_tissue_area_bound_box_relaxed,
        'tissue_area_bounding_box_peak_contraction': avg_tissue_area_bound_box_peak_contraction,
        'total_number_of_sarcomeres': avg_number_of_sarcomeres,
        'sarcomere_density_convex_hull': sarcomere_density_convext_hull,
        'sarcomere_density_bounding_box': sarcomere_density_bound_box,
        'noise_level': noise_level
    }

    return metrics


def extract_partition_metrics(
    sarcomere_data_df: pd.DataFrame,
    smoothed_signal_array: np.ndarray,
    analysis_results_dict: dict,
    PIXEL_TO_MICRON: float = 0.1625,
    FRAME_RATE: float = 1.0,
    FRAME_AVG_RANGE: int = 5
) -> Dict[str, float]:
    """
    Compute functional metrics for a single tissue partition.

    Parameters
    ----------
    sarcomere_data_df : pd.DataFrame
        Must contain columns ['frame','length'] (in px) and already be filtered
        to the partition of interest.
    smoothed_signal_array : np.ndarray
        GPR-smoothed mean sarcomere length per frame (in px).
    analysis_results_dict : dict
        Output of analyze_signal, with keys 'relaxed_states', 'contractions', 'expansions'.
    PIXEL_TO_MICRON : float
        Conversion from px to µm.
    FRAME_RATE : float
        Video frame rate (fps), for time conversions.
    FRAME_AVG_RANGE : int
        Number of frames around each peak to average.

    Returns
    -------
    metrics : dict
        Keys:
          - num_contractions
          - contraction_period (s)
          - contraction_frequency (Hz)
          - relaxed_sarcomere_length_mean, _median, _q25, _q75, _std (µm)
          - peak_sarcomere_length_mean, _median, _q25, _q75, _std (µm)
          - shortening_amplitude (µm)
          - peak_shortening_velocity, peak_lengthening_velocity (µm/s)
          - contraction_onset_to_relaxation_end_time (s)
          - contraction_onset_to_peak_contraction_time (s)
          - contraction_onset_to_50_contracted_time (s)
          - half_contracted_to_peak_contraction_time (s)
          - peak_contraction_to_relaxation_end_time (s)
          - peak_contraction_to_50_relaxed_time (s)
          - half_relaxed_to_full_relaxation_time (s)
          - total_number_of_sarcomeres
          - noise_level (µm)
    """
    # 1) Convert units
    df = sarcomere_data_df.copy()
    df['length'] *= PIXEL_TO_MICRON
    signal = smoothed_signal_array * PIXEL_TO_MICRON

    # 2) Pull out cycle info
    relaxed   = analysis_results_dict.get('relaxed_states', [])
    contractions = analysis_results_dict.get('contractions', [])
    relaxations  = analysis_results_dict.get('expansions', [])

    # 3) Basic counts
    N = min(len(relaxed), len(contractions), len(relaxations))
    if N == 0:
        return {k: np.nan for k in [
            'num_contractions','contraction_period','contraction_frequency',
            'relaxed_sarcomere_length_mean','relaxed_sarcomere_length_median',
            'relaxed_sarcomere_length_q25','relaxed_sarcomere_length_q75',
            'relaxed_sarcomere_length_std','peak_sarcomere_length_mean',
            'peak_sarcomere_length_median','peak_sarcomere_length_q25',
            'peak_sarcomere_length_q75','peak_sarcomere_length_std',
            'shortening_amplitude','peak_shortening_velocity',
            'peak_lengthening_velocity',
            'contraction_onset_to_relaxation_end_time',
            'contraction_onset_to_peak_contraction_time',
            'contraction_onset_to_50_contracted_time',
            'half_contracted_to_peak_contraction_time',
            'peak_contraction_to_relaxation_end_time',
            'peak_contraction_to_50_relaxed_time',
            'half_relaxed_to_full_relaxation_time',
            'total_number_of_sarcomeres','noise_level'
        ]}

    # 4) Time‐based periods & frequency
    starts_with_relaxed = relaxed[0]['start_idx'] < contractions[0]['start_idx']
    periods = []
    for i in range(N):
        if starts_with_relaxed:
            t0 = relaxed[i]['start_idx']
            t1 = relaxations[i]['end_idx']
        else:
            t0 = contractions[i]['start_idx']
            t1 = relaxed[i]['end_idx']
        periods.append((t1 - t0) / FRAME_RATE)
    contraction_period    = np.mean(periods)
    contraction_frequency = np.mean([1/p for p in periods])

    # 5) Per‐frame length stats
    stats = df.groupby('frame')['length'].agg(
        mean   = 'mean',
        median = 'median',
        std    = 'std',
        q25    = lambda x: np.percentile(x, 25),
        q75    = lambda x: np.percentile(x, 75)
    ).reset_index()

    def collect(idx_list, kind):
        arr = []
        for j in range(N):
            start = idx_list[j]['start_idx']
            end   = idx_list[j]['end_idx']
            if kind=='relaxed':
                sel = stats[(stats.frame>=start)&(stats.frame< end)]
            else:  # peak around contraction end
                peak = contractions[j]['end_idx']
                sel = stats[(stats.frame>=peak-FRAME_AVG_RANGE)&(stats.frame<=peak+FRAME_AVG_RANGE)]
            if not sel.empty:
                arr.append(sel[['mean','median','q25','q75','std']].values)
        return np.vstack(arr) if arr else np.empty((0,5))

    relaxed_vals = collect(relaxed,   'relaxed')
    peak_vals    = collect(contractions, 'peak')

    relaxed_mean, relaxed_median, relaxed_q25, relaxed_q75, relaxed_std = np.nanmean(relaxed_vals, axis=0)
    peak_mean,    peak_median,    peak_q25,    peak_q75,    peak_std    = np.nanmean(peak_vals,    axis=0)

    # 6) Shortening amplitude
    amps = []
    for j in range(N):
        rel = stats[(stats.frame>= relaxed[j]['start_idx']) & (stats.frame< relaxed[j]['end_idx'])]['mean']
        peak = stats[
            (stats.frame>= contractions[j]['end_idx']-FRAME_AVG_RANGE) &
            (stats.frame<= contractions[j]['end_idx']+FRAME_AVG_RANGE)
        ]['mean']
        if not rel.empty and not peak.empty:
            amps.append(rel.mean() - peak.mean())
    shortening_amplitude = np.mean(amps)

    # 7) Velocities
    vel = np.gradient(signal) * FRAME_RATE
    peak_short_vel = np.mean([ np.min(vel[c['start_idx']:c['end_idx']+1]) for c in contractions[:N] ])
    peak_relax_vel = np.mean([ np.max(vel[e['start_idx']:e['end_idx']+1]) for e in relaxations[:N] ])

    # 8) Timing metrics (in seconds)
    def timing(states, s_key, e_key):
        return np.mean([ (st[e_key]-st[s_key]) / FRAME_RATE for st in states[:N] ])

    onset_to_peak = timing(contractions, 'start_idx','end_idx')
    onset_to_end  = timing(relaxations,  'start_idx','end_idx')
    onset_to_50   = onset_to_peak * 0.5
    half_to_peak  = timing(contractions, 'end_idx','start_idx')
    peak_to_50rel = timing(relaxations,  'end_idx','start_idx')
    half_relax    = timing(relaxations,  'start_idx','end_idx') - peak_to_50rel

    # 9) Counts & noise
    total_sarcs = df.groupby('frame').size().mean()
    noise_level = np.nanstd(signal - stats.set_index('frame')['mean'].reindex_like(stats['frame']).values)

    return {
        'num_contractions':                        N,
        'contraction_period':                      contraction_period,
        'contraction_frequency':                   contraction_frequency,
        'relaxed_sarcomere_length_mean':           relaxed_mean,
        'relaxed_sarcomere_length_median':         relaxed_median,
        'relaxed_sarcomere_length_q25':            relaxed_q25,
        'relaxed_sarcomere_length_q75':            relaxed_q75,
        'relaxed_sarcomere_length_std':            relaxed_std,
        'peak_sarcomere_length_mean':              peak_mean,
        'peak_sarcomere_length_median':            peak_median,
        'peak_sarcomere_length_q25':               peak_q25,
        'peak_sarcomere_length_q75':               peak_q75,
        'peak_sarcomere_length_std':               peak_std,
        'shortening_amplitude':                    shortening_amplitude,
        'peak_shortening_velocity':                peak_short_vel,
        'peak_lengthening_velocity':               peak_relax_vel,
        'contraction_onset_to_relaxation_end_time':onset_to_end,
        'contraction_onset_to_peak_contraction_time':onset_to_peak,
        'contraction_onset_to_50_contracted_time': onset_to_50,
        'half_contracted_to_peak_contraction_time':half_to_peak,
        'peak_contraction_to_relaxation_end_time': peak_to_50rel + onset_to_end,  # adjust if needed
        'peak_contraction_to_50_relaxed_time':      peak_to_50rel,
        'half_relaxed_to_full_relaxation_time':     half_relax,
        'total_number_of_sarcomeres':               total_sarcs,
        'noise_level':                              noise_level,
    }