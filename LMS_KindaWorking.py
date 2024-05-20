import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def read_noise_image(noise_image_path):
    noise_image = cv2.imread(noise_image_path, cv2.IMREAD_GRAYSCALE)
    return noise_image

def estimate_local_variance(image, window_size=(7, 7)):
    padded_image = cv2.copyMakeBorder(image, *[(s - 1) // 2 for s in window_size] * 2, cv2.BORDER_REFLECT)
    local_variance = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded_image[i:i+window_size[0], j:j+window_size[1]]
            local_variance[i, j] = np.var(patch)
    return local_variance

def add_noise(original_image, noise_image):
    if original_image.shape != noise_image.shape:
        raise ValueError("Noise image and original image must have the same shape.")
    noise_image = noise_image * (original_image.max() - original_image.min()) / 255.0
    noisy_image = original_image + noise_image
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def lms_denoise(original_image, noisy_image, filter_size=(3, 3), step_size=0.01, num_iterations=10, convergence_threshold=1e-6):
    h, w = original_image.shape
    filter_height, filter_width = filter_size
    filter_weights = np.zeros(filter_size)

    for _ in range(num_iterations):
        predictions = np.zeros_like(original_image, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                # Extract the patch considering border cases
                i_min = max(0, i - filter_height // 2)
                i_max = min(h, i + filter_height // 2 + 1)
                j_min = max(0, j - filter_width // 2)
                j_max = min(w, j + filter_width // 2 + 1)
                patch = noisy_image[i_min:i_max, j_min:j_max]

                # Ensure the patch and filter weights have the same shape
                patch_height, patch_width = patch.shape
                if patch_height != filter_height or patch_width != filter_width:
                    continue

                # Compute the prediction using the patch and filter weights
                prediction = np.sum(patch * filter_weights)

                # Update the predictions array
                predictions[i, j] = prediction

        # Compute the errors
        errors = noisy_image - predictions

        # Update filter weights
        filter_updates = np.zeros_like(filter_weights, dtype=np.float64)

        for i in range(filter_height):
            for j in range(filter_width):
                # Extract the corresponding part of the errors array considering border cases
                i_min = max(0, i - h // 2)
                i_max = min(h, i + h // 2 + 1)
                j_min = max(0, j - w // 2)
                j_max = min(w, j + w // 2 + 1)
                error_patch = errors[i_min:i_max, j_min:j_max]

                # Compute the filter update
                filter_updates[i, j] = np.sum(error_patch * noisy_image[i_min:i_max, j_min:j_max])

        # Update the filter weights
        filter_weights += step_size * filter_updates

        # Check for convergence
        if np.linalg.norm(filter_updates) < convergence_threshold:
            break

    return reconstruct_denoised_image(noisy_image, filter_weights)

def reconstruct_denoised_image(noisy_image, filter_weights):
    patch_h, patch_w = filter_weights.shape
    padded_noisy_image = np.pad(noisy_image, ((patch_h // 2, patch_h // 2), (patch_w // 2, patch_w // 2)), mode='reflect')
    patches = np.lib.stride_tricks.sliding_window_view(padded_noisy_image, (patch_h, patch_w))
    denoised_image = np.tensordot(patches, filter_weights, axes=((2, 3), (0, 1)))
    return denoised_image[:noisy_image.shape[0], :noisy_image.shape[1]]  # Ensure the output has the correct shape

def adjust_filter_size(original_image, noisy_image, max_filter_size=(5, 5), step_size=0.01, num_iterations=10):
    best_psnr = 0
    best_filter_size = None

    for filter_size_h in range(1, max_filter_size[0] + 1):
        for filter_size_w in range(1, max_filter_size[1] + 1):
            filter_size = (filter_size_h, filter_size_w)
            denoised_image = lms_denoise(original_image, noisy_image, filter_size=filter_size, step_size=step_size, num_iterations=num_iterations)
            denoised_image = denoised_image.astype(original_image.dtype)  # Ensure same dtype before computing PSNR
            psnr_value = psnr(original_image, denoised_image)
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_filter_size = filter_size

    return best_filter_size, best_psnr

# Load the original image
original_image = cv2.imread('/home/ansor/Desktop/CodesMarch/image.jpg', cv2.IMREAD_GRAYSCALE)

# Load the noise image
noise_image = read_noise_image('/home/ansor/Desktop/CodesMarch/noise_image.jpg')

# Add noise to the original image
noisy_image = add_noise(original_image, noise_image)

# Estimate local variance of the noise image
local_variance = estimate_local_variance(noise_image)

# Adaptively adjust filter size based on local variance
adapted_filter_size, best_psnr = adjust_filter_size(original_image, noisy_image)

# Denoise the noisy image using LMS adaptive filtering with the adapted filter size
denoised_image = lms_denoise(original_image, noisy_image, filter_size=adapted_filter_size)

# Normalize the denoised image to the same intensity range as the original image
denoised_image = (denoised_image - denoised_image.min()) * (original_image.max() - original_image.min()) / (denoised_image.max() - denoised_image.min())
denoised_image = denoised_image.astype(original_image.dtype)

# Compute PSNR and SSIM
psnr_denoised = psnr(original_image, denoised_image)
ssim_denoised = ssim(original_image, denoised_image, data_range=denoised_image.max() - denoised_image.min())

# Display the results
print("PSNR (Peak Signal-to-Noise Ratio):")
print(f"Denoised: {psnr_denoised:.2f} dB")

print("\nSSIM (Structural Similarity Index):")
print(f"Denoised: {ssim_denoised:.2f}")

# Display the original, noisy, and denoised images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

axes[2].imshow(denoised_image, cmap='gray')
axes[2].set_title('Denoised Image')
axes[2].axis('off')

plt.show()