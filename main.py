import os, cv2, imutils, numpy as np

class DocumentPreprocessor:
    """
        Handles document scanner pipeline: preprocessing an image
        so its ready for contour/edge-based document detection.
    """

    def __init__(self, image_path:str, resize_height: int = 500):
        """
        Args:
            image_path (str): Path to the input image on disk.
            resize_height (int, optional): Target height (in px) to resize the image to 
            before processing. Smaller images process faster and Canny/Contour detection
            is more robust on a normalized scale.
             
            Defaults to 500.
        """
        self.image_path = image_path
        self.resize_height = resize_height

        self.original_image = None
        self.resized_image = None
        self.ratio = None
        self.gray_image = None
        self.blurred_image =None
        self.edged_image = None

    def load_image(self):
        """Load the input image from disk."""
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to read image (unsupported/corrupt file): {self.image_path}")

        return self

    def resize_image(self):
        """
        Resize the image down to `resize_height`, preserving aspect ratio. Stores the resize ratio (original/resized) so contours found on the resized image can be scaled back to original resolution later.
        """
        if self.original_image is None:
            raise RuntimeError("Call load_image() before resize_image().")

        original_height = self.original_image.shape[0]
        self.ratio = original_height / float(self.resize_height)

        self.resize_image = imutils.resize(self.original_image, height=self.resize_height)
        return self

    def convert_to_grayscale(self):
        """Convert the resized image to grayscale"""
        if self.resize_image is None:
            raise RuntimeError("Call resize_image() before convert_to_grayscale()")

        self.gray_image = cv2.cvtColor(self.resize_image, cv2.COLOR_BGR2GRAY)
        return self

    def apply_gaussian_blur(self, kernel_size: tuple = (5, 5), sigma: int = 0):
        """
        Apply Gaussian blur to the grayscale image to reduce noise.

        A (5,5) kernel is a common, well-tested choice for document scanning at ~500px working height: strong enough to suppress paper texture/noise without smearing away the document's edges. Both kernel dimensions must be positive and odd.
        """

        if self.gray_image is None:
            raise RuntimeError("Call convert_to_grayscale() before apply_gaussian_blur()")

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("Gaussian blur kernel dimensions must be odd numbers")

        self.blurred_image = cv2.GaussianBlur(self.gray_image, kernel_size, sigma)
        return self

    def apply_canny_edge_detection(self, low_threshold: int = 30, high_threshold: int = 150):
        """Apply Canny edge detection on the blurred grayscale image.
        
        75/200 is a commonly used starting point for document-edge detection under typical lighting - low enough to catch the document boundary, high enough to suppress background noise. Tune per-image if edges are too broken or too  noisy.
        """

        if self.blurred_image is None:
            raise RuntimeError("Call apply_gaussian_blur() before apply_canny_edge_detection().")

        self.edged_image = cv2.Canny(self, low_threshold, high_threshold)
        return self

    def save_edged_image(self, output_path: str):
        """Save the edge-detected image to disk."""
        if self.edged_image is None:
            raise RuntimeError("No edged image to save. Run the pipeline first.")

        cv2.imwrite(output_path, self.edged_image)
        print(f"[INFO] EDGE DETECTED IMAGE SAVED TO: {output_path}")
        return self

    def show_step(self, window_name: str, image: np.ndarray, wait: bool = True):
        """
        Display an image in a window (useful for visually confirming each preproccessing step, e.g. the document outline in the edge-detected image).
        """
        cv2.imshow(window_name, image)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows(window_name)
        return self

    def run(self, show:bool = False):
        """
        Args:
            show: If True, display each intermediate result in a window.
        """
        self.load_image()
        self.resize_image()
        self.convert_to_grayscale()
        self.apply_gaussian_blur()
        self.apply_canny_edge_detection()

        if show:
            self.show_step("Resized Image", self.resized_image)
            self.show_step("Grayscale", self.gray_image)
            self.show_step("Blurred", self.blurred_image)
            self.show_step("Edged (Canny)", self.edged_image)

        return self