import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class ImageEvaluator:
    def __init__(self):
        pass

    def calculate_psnr_ssim(self, real_image, generated_image):
        # Normalize images to range [0, 255]
        real_image = (real_image * 127.5 + 127.5).numpy().astype('uint8')
        generated_image = (generated_image * 127.5 + 127.5).numpy().astype('uint8')

        # Calculate PSNR
        psnr_value = tf.image.psnr(real_image, generated_image, max_val=255)

        # Calculate SSIM
        ssim_value = ssim(real_image, generated_image, multichannel=True, data_range=255)

        return psnr_value, ssim_value

    def display_images(self, real_A, generated_B, real_B, generated_A):
        plt.figure(figsize=(10, 10))
        titles = ['Real A', 'Generated B', 'Real B', 'Generated A']
        images = [real_A[0], generated_B[0], real_B[0], generated_A[0]]

        for i, image in enumerate(images):
            plt.subplot(2, 2, i + 1)
            plt.title(titles[i])
            plt.imshow(image * 0.5 + 0.5)
            plt.axis('off')

            if i == 1:  # Assuming generated_B is the comparison for real_A
                psnr, ssim_score = self.calculate_psnr_ssim(real_A[0], generated_B[0])
                plt.xlabel(f"PSNR: {psnr:.2f}, SSIM: {ssim_score:.3f}")

            if i == 3:  # Assuming generated_A is the comparison for real_B
                psnr, ssim_score = self.calculate_psnr_ssim(real_B[0], generated_A[0])
                plt.xlabel(f"PSNR: {psnr:.2f}, SSIM: {ssim_score:.3f}")

        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    evaluator = ImageEvaluator()
    # Assuming `real_A`, `generated_B`, `real_B`, `generated_A` are preloaded TensorFlow tensors or numpy arrays.
    # evaluator.display_images(real_A, generated_B, real_B, generated_A)
