import numpy as np
import pandas as pd
import math
import multiprocessing
import os

from pathlib import Path

from sarcgraph import SarcGraph


def detect_and_filter_sarcomeres_in_frame(
    img_frame: np.ndarray, # Corrected: Direct arguments
    sarcgraph_instance: SarcGraph, # Corrected: Direct arguments
    frame_index: int # Corrected: Direct arguments
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detects Z-discs and sarcomeres in a single image frame using SarcGraph,
    and applies filtering based on myofibril properties.

    Args:
        img_frame (np.ndarray): The 2D grayscale image frame.
        sarcgraph_instance (SarcGraph): An initialized SarcGraph object.
        frame_index (int): The index of the current frame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames:
                                           (zdiscs_df, sarcomeres_df).
                                           Returns empty DataFrames with correct columns
                                           if no valid sarcomeres/z-discs are found.
    """
    # Define consistent empty DataFrames with correct column schemas
    empty_zdiscs_df_schema = pd.DataFrame(columns=['frame', 'x', 'y', 'p1_x', 'p1_y', 'p2_x', 'p2_y'])
    empty_sarcomeres_df_schema = pd.DataFrame(columns=['frame', 'sarc_id', 'x', 'y', 'length', 'width', 'angle', 'zdiscs'])

    # 1. Z-disc Segmentation
    zdiscs_df = sarcgraph_instance.zdisc_segmentation(raw_frames=img_frame)

    # If no z-discs are detected, no sarcomeres can exist. Return both as empty.
    if zdiscs_df.empty:
        final_empty_zdiscs = empty_zdiscs_df_schema.copy()
        final_empty_zdiscs['frame'] = frame_index
        final_empty_sarcomeres = empty_sarcomeres_df_schema.copy()
        final_empty_sarcomeres['frame'] = frame_index
        return final_empty_zdiscs, final_empty_sarcomeres

    # 2. Sarcomere Detection
    sarcomeres_df, myofibrils_graphs = sarcgraph_instance.sarcomere_detection(
        segmented_zdiscs=zdiscs_df
    )

    # If z-discs were found but no sarcomeres were detected, return the detected z-discs and an empty sarcomeres DataFrame.
    if sarcomeres_df.empty:
        zdiscs_df['frame'] = frame_index
        final_empty_sarcomeres = empty_sarcomeres_df_schema.copy()
        final_empty_sarcomeres['frame'] = frame_index
        return zdiscs_df, final_empty_sarcomeres

    # 3. Filter sarcomeres and z-discs based on myofibril alignment
    valid_zdisc_particles = set()
    for myo_graph in myofibrils_graphs:
        # Only consider myofibrils that have at least one sarcomere (i.e., at least 2 Z-discs connected)
        if len(myo_graph.edges) == 0:
            continue
    
        end_nodes = [node for node, degree in myo_graph.degree() if degree == 1]

        if len(end_nodes) == 2: # Ensure it's a linear segment with clear ends for angle calculation
            z1_particle_id, z2_particle_id = end_nodes
            
            # Retrieve coordinates carefully, handling potential missing particles
            p1_coords = zdiscs_df[zdiscs_df.particle == z1_particle_id][['y', 'x']]
            p2_coords = zdiscs_df[zdiscs_df.particle == z2_particle_id][['y', 'x']]

            if not p1_coords.empty and not p2_coords.empty:
                p1 = p1_coords.iloc[0].values
                p2 = p2_coords.iloc[0].values

                # Calculate angle in degrees
                angle = np.abs(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi)
                
                # Original code's angle filter: skips if angle is between 60 and 120 degrees.
                # This means it *excludes* myofibrils that are roughly vertical.
                if angle > 60 and angle < 120:
                    continue # Skip to the next myofibril, effectively filtering out its Z-discs/sarcomeres

        # If the myofibril passed the filters, add all z-discs from this myofibril to the set of valid ones
        for z1_edge, z2_edge in myo_graph.edges:
            valid_zdisc_particles.add(z1_edge)
            valid_zdisc_particles.add(z2_edge)

    # Filter z-discs and sarcomeres based on the collected valid_zdisc_particles
    zdiscs_df_filtered = zdiscs_df[zdiscs_df.particle.isin(valid_zdisc_particles)].copy()
    
    # Filter sarcomeres: both of its z-discs must be in the valid set
    sarcomeres_df_filtered = sarcomeres_df[
        sarcomeres_df.apply(
            lambda row: int(row.zdiscs.split(',')[0]) in valid_zdisc_particles and \
                        int(row.zdiscs.split(',')[1]) in valid_zdisc_particles,
            axis=1
        )
    ].copy()

    # Add frame index to filtered DataFrames
    zdiscs_df_filtered['frame'] = frame_index
    sarcomeres_df_filtered['frame'] = frame_index

    return zdiscs_df_filtered, sarcomeres_df_filtered


def process_all_frames_for_sarcomeres(
    all_frames: list[np.ndarray],
    sarcgraph_instance: SarcGraph,
    save_filename: str,
    num_processes: int | None = None
):
    """
    Processes all frames in parallel to detect and filter sarcomeres and Z-discs,
    then saves the aggregated results to a compressed NumPy file.

    Args:
        all_frames (list[np.ndarray]): List of image frames to process.
        sarcgraph_instance (SarcGraph): An initialized SarcGraph object.
        save_filename (str): Path to the file where results will be saved (e.g., .npz format).
        num_processes (int, optional): Number of processes to use for parallel processing.
                                        Defaults to the number of CPU cores.
    """
    if num_processes is None:
        num_processes = os.cpu_count()
    if num_processes is None: # Fallback for systems where os.cpu_count() might return None
        num_processes = 1

    print(f"INFO: Using {num_processes} processes for parallel sarcomere detection.")

    # Prepare arguments for multiprocessing: (img_frame, sarcgraph_instance, frame_index)
    # sarcgraph_instance is passed to each child process. It should be pickleable.
    # SarcGraph objects generally are.
    task_args = [
        (all_frames[i], sarcgraph_instance, i) for i in range(len(all_frames))
    ]

    all_zdiscs_results = []
    all_sarcomeres_results = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use starmap for functions that take multiple arguments unpacked from a tuple
        results_per_frame = pool.starmap(detect_and_filter_sarcomeres_in_frame, task_args)

    # Aggregate results from all frames
    for frame_index, (zdiscs_df, sarcomeres_df) in enumerate(results_per_frame):
        # Only append if DataFrames are not empty (or have valid data)
        # Empty DataFrames are handled by detect_and_filter_sarcomeres_in_frame
        all_zdiscs_results.append(zdiscs_df)
        all_sarcomeres_results.append(sarcomeres_df)

        if (frame_index + 1) % 200 == 0:
            print(f"INFO: Aggregated results for frame {frame_index + 1}/{len(all_frames)}")

    # Concatenate all results into final DataFrames
    # Use pd.concat and handle empty list to avoid ValueError
    final_zdiscs_df = pd.concat(all_zdiscs_results, ignore_index=True) if all_zdiscs_results else pd.DataFrame()
    final_sarcomeres_df = pd.concat(all_sarcomeres_results, ignore_index=True) if all_sarcomeres_results else pd.DataFrame()

    # Store pandas DataFrames
    zdiscs_filename = Path(save_filename).parent / f"{Path(save_filename).stem}_zdiscs.csv"
    sarcomeres_filename = Path(save_filename).parent / f"{Path(save_filename).stem}_sarcomeres.csv"
    
    final_zdiscs_df.to_csv(zdiscs_filename, index=False)
    final_sarcomeres_df.to_csv(sarcomeres_filename, index=False)

    print(f"INFO: Aggregated Z-disc and Sarcomere data saved to: {save_filename}")