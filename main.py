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
        self.contours = None
        self.document_contour = None
        self.contour_preview_image = None

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

        self.resized_image = imutils.resize(self.original_image, height=self.resize_height)
        return self

    def convert_to_grayscale(self):
        """Convert the resized image to grayscale"""
        if self.resized_image is None:
            raise RuntimeError("Call resize_image() before convert_to_grayscale()")

        self.gray_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
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

        self.edged_image = cv2.Canny(self.blurred_image, low_threshold, high_threshold)
        return self

    def save_edged_image(self, output_path: str):
        """Save the edge-detected image to disk."""
        if self.edged_image is None:
            raise RuntimeError("No edged image to save. Run the pipeline first.")

        cv2.imwrite(output_path, self.edged_image)
        print(f"[INFO] EDGE DETECTED IMAGE SAVED TO: {output_path}")
        return self

    
    def find_contour(self, num_candidates: int = 5):
        """Find contours on the edge detected image.
        
        Args:
        num_candidates: It indicates the number of elements I have ,
        Suppose Image contains:
        - Paper
        - Keyboard
        - mouse
        - coffee cup
        - chair
        - pen
        - charger

        It will only detect the top 5 contours. 
        """
        if self.edged_image is None:
            raise RuntimeError("No edged image to save. Run the pipeline first.")

        contours, hierarchy = cv2.findContours(self.edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = sorted(contours, key=cv2.contourArea, reverse=True)

        
        epsilon_ratio = 0.02
        for contour in self.contours[:num_candidates]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
            if len(approx) == 4:
                self.document_contour = approx
                break


        if self.document_contour is None:
            x, y, w, h = cv2.boundingRect(self.contours[0])

            self.document_contour = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ], dtype=np.int32)
        
        return self    

    def draw_document_contour(self):
        if self.document_contour is None:
            raise RuntimeError("No contours is found, Call the find_contour first")

        # image_copy = self.original_image.copy()
        image_copy = self.resized_image.copy()


        contours_index = -1
        # The pixel coordinates of the contour points are listed in the obtained contours. Using this argument, you can specify the index position from this list, indicating exactly which contour point you want to draw. Providing a negative value will draw all the contour points.
        
        contour_color = (0, 255, 0)
        # BGR - BLUE DISABLE, GREEN ENABLE, RED DISABLE

        contour_thickness = 2
        # thickness of the contour

        cv2.drawContours(image_copy, [self.document_contour], contours_index, contour_color, contour_thickness)

        self.contour_preview_image = image_copy
        return self

    def show_step(self, window_name: str, image: np.ndarray, wait: bool = True):
        """
        Display an image in a window (useful for visually confirming each preproccessing step, e.g. the document outline in the edge-detected image).
        """
        cv2.imshow(window_name, image)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
        self.find_contour()
        self.draw_document_contour()

        if show:
            self.show_step("Resized Image", self.resized_image)
            self.show_step("Grayscale", self.gray_image)
            self.show_step("Blurred", self.blurred_image)
            self.show_step("Edged (Canny)", self.edged_image)
            self.show_step("Document Outline", self.contour_preview_image)

        return self


def main():
    input_path = "" # Place your image input path
    output_dir = "" # Place your image output path

    os.makedirs(output_dir, exist_ok=True)

    processor = DocumentPreprocessor(input_path)
    processor.run(show=False)  # set to True if you have a display and want live windows

    # Save each intermediate stage so you can inspect them afterward without a GUI
    cv2.imwrite(os.path.join(output_dir, "1_resized.jpg"), processor.resized_image)
    cv2.imwrite(os.path.join(output_dir, "2_gray.jpg"), processor.gray_image)
    cv2.imwrite(os.path.join(output_dir, "3_blurred.jpg"), processor.blurred_image)
    cv2.imwrite(os.path.join(output_dir, "4_edged.jpg"), processor.edged_image)
    cv2.imwrite(os.path.join(output_dir, "5_contour.jpg"), processor.contour_preview_image)

    print(f"[INFO] All stage outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()