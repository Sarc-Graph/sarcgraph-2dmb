import numpy as np
import pandas as pd
from pathlib import Path

def label_sarcomere_partitions(
    sarcomere_df: pd.DataFrame, 
    bounding_boxes: np.ndarray
) -> pd.DataFrame:
    """
    Adds 'horizontal_region' and 'vertical_region' columns to the sarcomere DataFrame
    based on the per-frame bounding box coordinates.

    Parameters:
    - sarcomere_df: DataFrame with columns ['frame', 'x', 'y', ...]
    - bounding_boxes: NumPy array of shape (num_frames, 4, 2) with box corner coords

    Returns:
    - A new DataFrame with two additional columns: 'horizontal_region' and 'vertical_region'.
    """
    # Ensure integer coordinates
    bbs = bounding_boxes.astype(np.int32)
    # Sort each frame's 4 points by x-coordinate
    idx_sort = np.argsort(bbs[:, :, 0], axis=1)
    bbs_sorted = np.take_along_axis(bbs, idx_sort[:, :, None], axis=1)

    # Copy input and prepare columns
    df = sarcomere_df.copy()
    df['horizontal_region'] = None
    df['vertical_region'] = None

    # Loop per frame
    for frame_num in np.unique(df['frame']):
        if frame_num < 0 or frame_num >= bbs_sorted.shape[0]:
            continue
        p1, p2, p3, p4 = bbs_sorted[int(frame_num)]
        # Define long and short axes
        upper = p1 if p1[1] < p2[1] else p2
        lower = p3 if p3[1] < p4[1] else p4
        long_axis = lower - upper
        short_axis = p2 - p1
        if short_axis[1] > 0:
            short_axis = -short_axis

        # Normalize
        long_norm = np.linalg.norm(long_axis)
        short_norm = np.linalg.norm(short_axis)
        if long_norm == 0 or short_norm == 0:
            continue
        long_hat = long_axis / long_norm
        short_hat = short_axis / short_norm

        # Get sarcomere positions for this frame
        mask = df['frame'] == frame_num
        positions = df.loc[mask, ['y', 'x']].values
        center = np.mean(bbs_sorted[int(frame_num)], axis=0)
        half_len = long_norm / 2

        # Project onto axes
        d_long = np.dot(positions - center, long_hat)
        d_short = np.dot(positions - center, short_hat)

        # Classify
        horiz = np.full(len(d_long), 'center', dtype=object)
        horiz[d_long > (half_len / 3)] = 'right'
        horiz[d_long < (-half_len / 3)] = 'left'
        vert = np.full(len(d_short), 'bottom', dtype=object)
        vert[d_short > 0] = 'top'

        # Assign back
        df.loc[mask, 'horizontal_region'] = horiz
        df.loc[mask, 'vertical_region'] = vert

    return df


def partition_sarcomere_data(
    sarcomere_csv: Path | str,
    bounding_box_npz: Path | str,
    output_csv: Path | str = None
) -> pd.DataFrame:
    """
    Loads sarcomere detection CSV and bounding-box NPZ, labels each row with partitions,
    and saves the annotated DataFrame to CSV.

    Parameters:
    - sarcomere_csv: Path to the detected-sarcomeres CSV file.
    - bounding_box_npz: Path to the .npz file containing 'boxes'.
    - output_csv: Path to save the annotated CSV. Defaults to overwrite sarcomere_csv.

    Returns:
    - The annotated DataFrame.
    """
    # Load inputs
    sarc_df = pd.read_csv(sarcomere_csv)
    data = np.load(bounding_box_npz, allow_pickle=True)
    boxes = data['boxes']  # shape: (num_frames, 4, 2)

    # Label partitions
    annotated = label_sarcomere_partitions(sarc_df, boxes)

    # Save
    out_path = Path(output_csv) if output_csv else Path(sarcomere_csv)
    annotated.to_csv(out_path, index=False)
    print(f"Saved partition-labeled sarcomere data to {out_path}")

    return annotated
