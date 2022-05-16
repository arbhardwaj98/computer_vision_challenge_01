import os
import cv2
import ray
import numpy as np


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize the pixel value of input image
    Normalization allows overcoming linear changes in pixel intensities
    :param img: Input image
    :return: Normalized image
    """
    # Normalize only along spatial dimensions
    spatial_axes = tuple(np.arange(img.ndim-1))

    # Compute the standard deviation of the pixel values for each channel
    img_std = np.std(img, axis=spatial_axes)

    # Normalize the image
    img_norm = img / img_std

    return img_norm


@ray.remote
def get_pixel_dist(
        query_img_path: str,
        normalized_reference_img: np.ndarray,
        dist_func: str = "dice"
) -> float:
    """
    Find the mean pixel-wise distance between the query and the reference images
    :param query_img_path: Path to query image
    :param normalized_reference_img: Normalized reference image
    :param dist_func: Function to use for calculating pixel-wise distance
    :return: Mean pixel-wise distance
    """

    # Load query image as a numpy array
    query_img = cv2.imread(os.path.join(query_image_dir, query_img_path))

    # Remove blank pixels, i.e. pixels with no intensity on all channels
    mask = np.nonzero(query_img.sum(axis=2))

    # Normalize only the non-zero pixels
    normalized_query_img = normalize_image(query_img[mask])

    # Find pixel-wise distance
    if dist_func == "error":  # Normalized error loss
        pixel_dist = (
            abs(normalized_reference_img[mask] - normalized_query_img)
            /
            (normalized_reference_img[mask] * normalized_query_img + 1)
        ).mean()

    elif dist_func == "dice":  # Dice loss
        pixel_dist = 1 - 2*(
                (normalized_reference_img[mask] * normalized_query_img)
                /
                (normalized_reference_img[mask] + normalized_query_img)
        ).mean()

    else:
        raise NotImplementedError

    return pixel_dist


if __name__ == "__main__":
    """
    ALGORITHM:
    
    This script loads the reference image and iterates over 
    the query images to find the best match. The best match
    corresponds to the image with the least pixel-wise distance.
       
    A mask is created to remove the blank pixels from the query
    image. Pixel values are compared only where query image
    pixels are not blank. The pixel values of the images are 
    first normalized to account for affine intensity changes. 
    The function get_pixel_dist finds the mean pixel-wise distance
    between the normalized reference and normalized query image. 
    """

    # Initialize ray for multiprocessing
    ray.init()

    # Outputs overlaid image if set to True
    output_overlay = True

    # Load and normalize the reference image
    reference_img = cv2.imread("reference_image/aerial.png")
    normalized_reference_img = normalize_image(reference_img)

    # Load paths to query images
    query_image_dir = "query_images"
    query_img_paths = os.listdir(query_image_dir)

    # Create a list of ray futures
    result_ids = [get_pixel_dist.remote(query_path, normalized_reference_img) for query_path in query_img_paths]

    # Accumulate results from ray futures
    results = ray.get(result_ids)

    # Find the image with minimum pixel-wise distance
    best_match_idx = np.argmin(np.array(results))
    print(f"Best matching image is ", query_img_paths[best_match_idx])

    # Save output image
    if output_overlay:
        output_path = "best_match_overlay.png"
        best_match = cv2.imread(os.path.join(query_image_dir, query_img_paths[best_match_idx]))
        output_img = cv2.addWeighted(reference_img, 0.5, best_match, 0.5, 0.0)
        cv2.imwrite(output_path, output_img)
        print(f"Best matching image is saved as", output_path)

