import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage.filters import threshold_otsu
import multiprocessing

def process_frame_bounding_box_hull(
    img_frame: np.ndarray,
    noise_region_height: int = 30,
    noise_std_multiplier: float = 1.5,
    gaussian_blur_kernel_size: int = 21
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Processes a single frame to extract bounding box and convex hull.

    Args:
        img_frame (np.ndarray): The input image frame.
        noise_region_height (int): Height of the region at the top of the frame assumed to contain only noise. Defaults to 30 pixels.
        noise_std_multiplier (float): Multiplier for noise standard deviation to set the noise suppression threshold. Defaults to 1.5.
        gaussian_blur_kernel_size (int): Size of the Gaussian blur kernel. Must be odd. Defaults to 21.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: A tuple containing the bounding box (box) and
                                                      convex hull points (hull_points) as NumPy arrays,
                                                      or (None, None) if no hull could be computed or
                                                      an error occurred during processing.
    """
    # Ensure kernel size is odd for cv2.GaussianBlur
    if gaussian_blur_kernel_size % 2 == 0:
        print(f"WARNING: Gaussian blur kernel size {gaussian_blur_kernel_size} is even. "
              f"Adjusting to {gaussian_blur_kernel_size + 1} for cv2.GaussianBlur.")
        gaussian_blur_kernel_size += 1

    try:
        # Extract noise statistics from the histogram
        # Ensure noise_region_height doesn't exceed image height
        effective_noise_height = min(noise_region_height, img_frame.shape[0])
        noise_mean = np.mean(img_frame[0:effective_noise_height, :])
        noise_std = np.std(img_frame[0:effective_noise_height, :])

        # Define a noise threshold (Mean + k*StdDev)
        noise_threshold = noise_mean + noise_std_multiplier * noise_std

        # Suppress noise by setting pixels below the threshold to zero
        denoised_image = np.copy(img_frame)
        denoised_image[img_frame < noise_threshold] = 0  # Keep only signal

        # Apply Gaussian blur to smooth the image
        denoised_image = cv2.GaussianBlur(denoised_image,
                                          (gaussian_blur_kernel_size, gaussian_blur_kernel_size),
                                          0)

        # Robust normalization to 0-255 range
        min_val = np.min(denoised_image)
        max_val = np.max(denoised_image)
        if max_val - min_val > 0: # Avoid division by zero
            denoised_image = (denoised_image - min_val) / (max_val - min_val) * 255
        else:
            # Image is flat (all same pixel value), no signal detected after denoising/blurring
            print(f"INFO: Frame is flat after denoising/blurring (min_val={min_val}, max_val={max_val}). Returning None, None.")
            return None, None # No meaningful signal to process further

        denoised_image = denoised_image.astype(np.uint8)

        # Apply Otsu thresholding
        local_otsu = threshold_otsu(denoised_image)

        # Convert to binary image
        binary_local_otsu = denoised_image >= local_otsu

        # Get coordinates of all white pixels. np.where returns (row, col) indices.
        white_pixel_indices = np.column_stack(np.where(binary_local_otsu > 0))

        # Convert indices to (x, y) coordinates.
        # Note: row -> y, col -> x
        points = white_pixel_indices[:, [1, 0]]

        if len(points) >= 3:  # ConvexHull requires at least 3 points
            # Compute the convex hull of the points
            hull = ConvexHull(points)

            # Extract the hull vertices
            hull_points = points[hull.vertices]

            rect = cv2.minAreaRect(hull_points)  # Returns (center, (width, height), angle)
            box = cv2.boxPoints(rect)  # Get 4 corner points
            box = np.intp(box)

            return box, hull_points
        else:
            print("INFO: Not enough points (less than 3) to compute Convex Hull for this frame.")
            return None, None

    except Exception as e:
        # Catch any unexpected errors during processing of this single frame
        print(f"ERROR: An unexpected error occurred while processing frame for bounding box/hull: {e}")
        # Optionally, you could print traceback for more detailed debugging during development:
        # import traceback
        # traceback.print_exc()
        return None, None


def process_frames_parallel_save_bbox_hull(all_frames: list[np.ndarray], save_filename: str, num_processes: int | None = None):
    """
    Processes frames in parallel to extract bounding boxes and convex hulls, and saves the results to a file.

    Args:
        all_frames (list[np.ndarray]): List of image frames to process.
        save_filename (str): Path to the file where results will be saved (e.g., .npz format).
        num_processes (int, optional): Number of processes to use for parallel processing. Defaults to the number of CPU cores.
    """
    num_frames = len(all_frames)
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"INFO: Using {num_processes} processes for parallel frame processing and saving to: {save_filename}")

    frame_data = [] # List to store results for each frame

    with multiprocessing.Pool(processes=num_processes) as pool:
        frame_results = pool.map(process_frame_bounding_box_hull, all_frames)

    for frame_index, (box, hull_points) in enumerate(frame_results):
        if (frame_index + 1) % 200 == 0:
            print(f"INFO: Processed frame {frame_index + 1}/{num_frames}")
        frame_data.append({'frame_index': frame_index, 'box': box, 'hull_points': hull_points})

    # Save results to file
    np.savez_compressed(
        save_filename,
        boxes=np.array([f['box'] if f['box'] is not None else np.array([]) for f in frame_data], dtype=object),
        hull_points=np.array([f['hull_points'] if f['hull_points'] is not None else np.array([]) for f in frame_data], dtype=object)
    )
    print(f"INFO: Bounding box and convex hull data saved to: {save_filename}")