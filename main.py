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
    spatial_axes = tuple(np.arange(img.ndim-1))  # Normalize only along spatial dimensions
    img_std = np.std(img, axis=spatial_axes)
    img_norm = img / img_std

    return img_norm


@ray.remote
def get_pixel_dist(query_img_path, normalized_reference_img, dist_func="dice"):

    # Load query image as a numpy array
    query_img = cv2.imread(os.path.join(query_image_dir, query_img_path))

    # Remove blank pixels, i.e. pixels with no intensity on all channels
    mask = np.nonzero(query_img.sum(axis=2))

    # Normalize only the non-zero pixels
    normalized_query_img = normalize_image(query_img[mask])

    # Find pixel-wise distance
    if dist_func == "error":  # Normalized error loss
        pixel_dist_norm = (
            abs(normalized_reference_img[mask] - normalized_query_img)
            /
            (normalized_reference_img[mask] * normalized_query_img + 1)
        ).mean()

    elif dist_func == "dice":  # Dice loss
        pixel_dist_norm = 1 - 2*(
                (normalized_reference_img[mask] * normalized_query_img)
                /
                abs(normalized_reference_img[mask] + normalized_query_img)
        ).mean()

    else:
        raise NotImplementedError

    return pixel_dist_norm


if __name__ == "__main__":

    ray.init()

    output_overlay = True  # Output overlaid image

    # Load and normalize the reference image
    reference_img = cv2.imread("reference_image/aerial.png")
    normalized_reference_img = normalize_image(reference_img)

    # Load paths to query images
    query_image_dir = "query_images"
    query_img_paths = os.listdir(query_image_dir)

    result_ids = [get_pixel_dist.remote(query_path, normalized_reference_img) for query_path in query_img_paths]
    results = ray.get(result_ids)

    best_match_idx = np.argmin(np.array(results))
    print(f"Best matching image is ", query_img_paths[best_match_idx])

    if output_overlay:
        output_path = "best_match_overlay.png"
        best_match = cv2.imread(os.path.join(query_image_dir, query_img_paths[best_match_idx]))
        output_img = cv2.addWeighted(reference_img, 0.5, best_match, 0.5, 0.0)
        cv2.imwrite(output_path, output_img)
        print(f"Best matching image is saved as", output_path)

