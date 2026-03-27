import numpy as np
from typing import List, Dict, Union
import scipy.ndimage as ndimage


def apply_temporal_smoothing(
    pose_results: List[Dict[str, np.ndarray]],
    window_size: int = 7,
    sigma: float = 1.5,
    min_score_threshold: float = 0.2
) -> List[Dict[str, np.ndarray]]:
    """Apply temporal smoothing to keypoint sequences for video stability.

    Args:
        pose_results: List of pose results for each frame
        window_size: Size of the Gaussian smoothing window (odd number)
        sigma: Standard deviation for Gaussian kernel
        min_score_threshold: Minimum confidence score to consider a keypoint valid

    Returns:
        List of smoothed pose results
    """
    if len(pose_results) < 3:
        # Not enough frames to smooth
        return pose_results

    # Ensure window size is odd
    window_size = max(3, window_size if window_size %
                      2 == 1 else window_size + 1)

    # Extract keypoints and scores across frames
    all_keypoints = []
    all_scores = []

    for frame_result in pose_results:
        all_keypoints.append(frame_result.get('keypoints', np.array([])))
        all_scores.append(frame_result.get('keypoints_scores', np.array([])))

    # Find maximum number of people across all frames
    max_people = max([kpts.shape[0] if kpts.size >
                     0 else 0 for kpts in all_keypoints])
    if max_people == 0:
        return pose_results  # No keypoints to smooth

    # Get dimensions
    num_frames = len(all_keypoints)
    num_keypoints = all_keypoints[0].shape[1] if all_keypoints[0].size > 0 else 0

    if num_keypoints == 0:
        return pose_results  # No keypoints to smooth

    # Create aligned arrays for smoothing, padding with zeros for missing people
    aligned_keypoints = np.zeros((num_frames, max_people, num_keypoints, 2))
    aligned_scores = np.zeros((num_frames, max_people, num_keypoints))

    # Fill in the data
    for i, (kpts, scores) in enumerate(zip(all_keypoints, all_scores)):
        if kpts.size > 0:
            aligned_keypoints[i, :kpts.shape[0]] = kpts
            aligned_scores[i, :scores.shape[0]] = scores

    # Create a mask for valid keypoints (above threshold)
    valid_mask = aligned_scores > min_score_threshold

    # Smooth each person's keypoints over time
    smoothed_keypoints = aligned_keypoints.copy()

    # For each person and keypoint
    for person_idx in range(max_people):
        for kpt_idx in range(num_keypoints):
            # Only smooth if we have enough valid points
            person_kpt_mask = valid_mask[:, person_idx, kpt_idx]
            if np.sum(person_kpt_mask) > window_size // 2:
                # Smooth X coordinates
                x_values = aligned_keypoints[:, person_idx, kpt_idx, 0]
                x_valid = np.where(person_kpt_mask, x_values, np.nan)

                # Replace NaN with interpolated values for smoothing
                x_interp = np.copy(x_valid)
                mask = np.isnan(x_interp)
                x_interp[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(~mask),
                    x_interp[~mask]
                )

                # Apply Gaussian filter
                x_smoothed = ndimage.gaussian_filter1d(
                    x_interp, sigma=sigma, truncate=2.0)

                # Only update valid keypoints
                smoothed_keypoints[:, person_idx, kpt_idx, 0] = np.where(
                    person_kpt_mask, x_smoothed, aligned_keypoints[:,
                                                                   person_idx, kpt_idx, 0]
                )

                # Smooth Y coordinates similarly
                y_values = aligned_keypoints[:, person_idx, kpt_idx, 1]
                y_valid = np.where(person_kpt_mask, y_values, np.nan)

                y_interp = np.copy(y_valid)
                mask = np.isnan(y_interp)
                y_interp[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(~mask),
                    y_interp[~mask]
                )

                y_smoothed = ndimage.gaussian_filter1d(
                    y_interp, sigma=sigma, truncate=2.0)

                smoothed_keypoints[:, person_idx, kpt_idx, 1] = np.where(
                    person_kpt_mask, y_smoothed, aligned_keypoints[:,
                                                                   person_idx, kpt_idx, 1]
                )

    # Update the original pose results with smoothed keypoints
    smoothed_results = []
    for i, result in enumerate(pose_results):
        new_result = result.copy()
        num_people = all_keypoints[i].shape[0] if all_keypoints[i].size > 0 else 0

        if num_people > 0:
            new_result['keypoints'] = smoothed_keypoints[i, :num_people].copy()

        smoothed_results.append(new_result)

    return smoothed_results
