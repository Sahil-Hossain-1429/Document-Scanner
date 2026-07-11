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
        self.destination_points = None
        self.max_width = None
        self.max_height = None
        self.scaled_points = None
        self.transform_matrix = None
        self.warped_image = None
        self.oriented_image = None

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

        draw_pts = self.document_contour.astype("int32").reshape(-1, 1, 2)
        # -1 → infer the number of points automatically
        # 1 → one coordinate record per point (required by many OpenCV APIs)
        # 2 → each point has two values: (x, y)
        cv2.drawContours(image_copy, [draw_pts], contours_index, contour_color, contour_thickness)

        self.contour_preview_image = image_copy
        return self

    def order_points(self):
        """
             Take the 4-point document contour (in arbitrary order) and reorder the points consistently as: top-left, top-right, bottom-right, bottom-left.

             Sum/diff trick:
             - top-left has the smallest sum value (x+y)
             - bottom-right has the largest value (x+y)
             - top-right has largest difference value (x-y)
             - bottom-left has smallest difference value (x-y)

             and the distance of these values are calculated from (0,0) coordinates

             Like this:

             (0,0)
              ____________________
             |
             |      A           B
             |
             |  
             |      C           D            
             |
        """
        if self.document_contour is None:
            raise RuntimeError("No contour found, Call find_contour() before order_points()")

        # document_contour is shape(4,1,2) from approxPolyDP / boundingRect -> reshape(4,2)
        pts = self.document_contour.reshape(4,2).astype("float32")

        ordered = np.zeros((4,2), dtype="float32")

        pointSum = pts.sum(axis=1)
        pointDifference = np.diff(pts,axis = 1).flatten()

        ordered[0] = pts[np.argmin(pointSum)]           #top-left
        ordered[2] = pts[np.argmax(pointSum)]           #bottom-right
        ordered[1] = pts[np.argmax(pointDifference)]    #top-right
        ordered[3] = pts[np.argmin(pointDifference)]    #bottom-left

        self.document_contour = ordered
        return self

    def compute_destination_rectangle(self):
        """
        Given the ordered 4 points (tl, tr, br, bl), compute the width and height of the "flattened" output image, and build the destination points array representing a perfect rectangle.

        width/height are each the MAX of their two corresponding edges, since the document may appear skewed/rotated in the source image and the two parallel edges won't measure exactly equal
        """

        if self.document_contour is None:
            raise RuntimeError("Call order_points() before compute_destination_rectangle().")

        (top_left, top_right, bottom_right, bottom_left) = self.document_contour


        # Width: distance along the top edge vs. the bottom edge
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        max_width = max(int(width_top), int(width_bottom))

        # Height: distance along the left edge vs. the right edge
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        max_height = max(int(height_left), int(height_right))

        # Destination rectangle: perfect top-down view, same corner order
        # as ordered_points (top_left, top_right, bottom_right, bottom_left)
        self.destination_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")        

        self.max_width = max_width
        self.max_height = max_height

    def scale_points_to_original(self):
        """
            Scale the ordered source points (found on the resized image) back up to full resolution coordinates, using the store resize ratio

            This lets us apply the perspective warp to the original, full quality image instead of the small working copy used for edge/contour detection
        """
        if self.document_contour is None:
            raise RuntimeError("Call order_points() before scale_points_to_original().")
        if self.ratio is None:
            raise RuntimeError("Call resize_image() before scale_points_to_original().")

        self.scaled_points = self.document_contour * self.ratio
        return self

    def compute_perspective_transform(self):
        """
            Compute the perspective transform matrix mapping the scaled source points (full resolution corners of the document) to the destination rectangle (a perfect top down view)
        """
        if self.scaled_points is None:
            raise RuntimeError("Call scale_points_to_original() before compute_perspective_transform().")
        if self.destination_points is None:
            raise RuntimeError("Call compute_destination_rectangle() before compute_perspective_transform().")

        self.transform_matrix = cv2.getPerspectiveTransform(
            self.scaled_points.astype("float32"),
            self.destination_points
        )

        return self

    def warp_image(self):
        """
            Apply the perspective transform to the full resolution original image,
            producing a flattened, straight on view of the document
        """
        if self.transform_matrix is None:
            raise RuntimeError("Call compute_perspective_transform() before warp_image().")
        if self.original_image is None:
            raise RuntimeError("No original image loaded.")

        self.warped_image = cv2.warpPerspective(
            self.original_image,
            self.transform_matrix,
            (self.max_width, self.max_height)
        )
        return self

    def save_wraped_image(self, output_path: str):
        """
            Save the final flattened/wrpaed document image to disk.
        """
        if self.warped_image is None:
            raise RuntimeError("No warped image to save. Run warp_image() first.")

        cv2.imwrite(output_path, self.warped_image)
        print(f"[INFO] WARPED (FLATTENED) IMAGE SAVED TO: {output_path}")
        return self

    def fix_orientation(self):
        """
        Rotate the warped image 90 degrees clockwise, then flip it vertically.

        cv2.ROTATE_90_CLOCKWISE rotates the image; cv2.flip(..., 0) flips
        around the horizontal axis (i.e. top-to-bottom / vertical flip).
        """
        if self.warped_image is None:
            raise RuntimeError("Call warp_image() before fix_orientation().")

        rotated = cv2.rotate(self.warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.oriented_image = cv2.flip(rotated, 0)  # 0 = flip vertically

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

    def run(self, show: bool = False):
        self.load_image()
        self.resize_image()
        self.convert_to_grayscale()
        self.apply_gaussian_blur()
        self.apply_canny_edge_detection()
        self.find_contour()
        self.order_points()
        self.compute_destination_rectangle()
        self.scale_points_to_original()
        self.compute_perspective_transform()
        self.warp_image()
        self.fix_orientation()          
        self.draw_document_contour()

        if show:
            self.show_step("Resized Image", self.resized_image)
            self.show_step("Grayscale", self.gray_image)
            self.show_step("Blurred", self.blurred_image)
            self.show_step("Edged (Canny)", self.edged_image)
            self.show_step("Document Outline", self.contour_preview_image)
            self.show_step("Warped (Flattened)", self.warped_image)
            self.show_step("Final Oriented", self.oriented_image)   

        return self


def main():
    input_path = "images/input/TestImage4.jpg" # Place your image input path
    output_dir = "images/output" # Place your image output path

    os.makedirs(output_dir, exist_ok=True)

    processor = DocumentPreprocessor(input_path)
    processor.run(show=False)  # set to True if you have a display and want live windows

    # Save each intermediate stage so you can inspect them afterward without a GUI
    cv2.imwrite(os.path.join(output_dir, "1_resized.jpg"), processor.resized_image)
    cv2.imwrite(os.path.join(output_dir, "2_gray.jpg"), processor.gray_image)
    cv2.imwrite(os.path.join(output_dir, "3_blurred.jpg"), processor.blurred_image)
    cv2.imwrite(os.path.join(output_dir, "4_edged.jpg"), processor.edged_image)
    cv2.imwrite(os.path.join(output_dir, "5_contour.jpg"), processor.contour_preview_image)
    cv2.imwrite(os.path.join(output_dir, "6_warped.jpg"), processor.warped_image)
    cv2.imwrite(os.path.join(output_dir, "7_final.jpg"), processor.oriented_image)

    print(f"[INFO] All stage outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()