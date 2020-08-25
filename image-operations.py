import cv2
import imutils
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_PLAIN
CAPTION_SCALE = 2
CAPTION_THICKNESS = 2


class Image:
    """
    Class for functions on image using OpenCV

    1.	Rotating an image
    2.	Smoothing an image
    3.	Counting objects
    4.	Converting an image to grayscale
    5.	Edge detection
    6.	Detecting and Drawing Contours
    """

    def __init__(self, image):

        width = image.shape[1]
        self.image = imutils.resize(image, int(width*0.7))
        self.height, self.width, self.depth = self.image.shape
        self.position = (20, self.height-10)

    def show(self, image=None, title=None):
        """function to display original or argument image

        Args:
            image (2D list): An image matrix. Defaults to None.
            title (String): To be shown in title bar. Defaults to None.
        """

        # show image on output window
        if title:
            title = title+": Press 0 to close window"
        else:
            title = "Press 0 to close window"
        if image is None:
            image = self.image.copy()
            # add text on image
            cv2.putText(image, 'Original Image',
                        self.position, FONT, CAPTION_SCALE,
                        BLACK, CAPTION_THICKNESS)
        cv2.imshow(title, image)
        cv2.waitKey(0)  # to avoid image from disappearing

    def rotate(self):
        """
        Rotating the image
        """

        original = self.image.copy()

        # rotate images about axis as center, with scale 0.5
        rotated_90 = imutils.rotate(
            original.copy(), angle=90, scale=0.5)  # 90 degrees
        rotated_m90 = imutils.rotate(
            original.copy(), angle=-90, scale=0.5)  # -90 degrees
        rotated_45 = imutils.rotate(
            original.copy(), angle=45, scale=0.5)  # 45 degrees
        rotated_m45 = imutils.rotate(
            original.copy(), angle=-45, scale=0.5)  # -45 degrees

        # add text on images
        cv2.putText(rotated_90, 'Rotated 90 degrees',
                    self.position, FONT, CAPTION_SCALE,
                    WHITE, CAPTION_THICKNESS)
        cv2.putText(rotated_m90, 'Rotated -90 degrees',
                    self.position, FONT, CAPTION_SCALE,
                    WHITE, CAPTION_THICKNESS)
        cv2.putText(rotated_45, 'Rotated 45 degrees',
                    self.position, FONT, CAPTION_SCALE,
                    WHITE, CAPTION_THICKNESS)
        cv2.putText(rotated_m45, 'Rotated -45 degrees',
                    self.position, FONT, CAPTION_SCALE,
                    WHITE, CAPTION_THICKNESS)

        # merge the 4 images together in a 2x2 matrix
        return np.concatenate(
            (np.concatenate(
                (imutils.resize(rotated_90, self.width//2),
                 imutils.resize(rotated_m90, self.width//2)), axis=1),
             np.concatenate(
                (imutils.resize(rotated_45, self.width//2),
                 imutils.resize(rotated_m45, self.width//2)), axis=1)), axis=0)

    def smoothen(self, method=None):
        """
        Smoothing the image
        """

        filter_size = (15, 15)  # size of the window for convolution

        original = self.image.copy()

        # using median blur to smoothen image
        blur = cv2.blur(original.copy(), filter_size)

        # using gaussian blur to smoothen image
        filter_sd = 0  # standard deviaion for filter
        gaussian = cv2.GaussianBlur(original.copy(), filter_size, filter_sd)

        # using median blur to smoothen image
        median = cv2.medianBlur(original.copy(), 15)

        # add text on images
        cv2.putText(original, 'Original Image',
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)
        cv2.putText(blur, 'Averaging Blur',
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)
        cv2.putText(gaussian, 'Gaussian Blur',
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)
        cv2.putText(median, 'Median Blur',
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)

        # merge the 4 images together in a 2x2 matrix
        return np.concatenate(
            (np.concatenate(
                (imutils.resize(original, self.width//2),
                 imutils.resize(blur, self.width//2)), axis=1),
             np.concatenate(
                (imutils.resize(gaussian, self.width//2),
                 imutils.resize(median, self.width//2)), axis=1)), axis=0)

    def detect_edges(self):
        """
        Edge detection using Canny algorithm
        """
        original = self.image.copy()

        # canny edge detection on image with threshold 70, 205
        edges = cv2.Canny(self.grayscale(), 70, 205)
        cv2.putText(edges, 'Edges',
                    self.position, FONT, CAPTION_SCALE,
                    WHITE, CAPTION_THICKNESS)
        return edges

    def grayscale(self):
        """
        Converting image to grayscale
        """

        # convert (blue, green, red) image to grayscale
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def fill_shapes(self, image):
        """
        Use thresholding to completely fill the shapes
        """

        # getting threshold values by trial and error
        return cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)[1]

    def draw_contours(self, image):
        """
        Detect contours and then draw on original image
        """

        # get contours from filled shapes image
        # CV_RETR_EXTERNAL: retrieves only the extreme outer contours
        # CV_CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal
        # segments and leaves only their end points.
        # For example, an up-right rectangular contour is encoded with 4 points
        contours = cv2.findContours(image,
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        objects = self.image.copy()

        # draw border around each contour
        for c in contours:
            cv2.drawContours(objects, [c], -1, (0, 0, 0), 3)

        # add text on image with count of objects
        caption = 'Contours. Number of objects: ' + str(len(contours))
        cv2.putText(objects, caption,
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)

        return objects

    def show_contours(self, gray, filled_shapes, objects):
        """
        function for showing entire detecting and highlighting procedure in
             a single image
        """

        original = self.image

        # convert to BRG to enable showing all images together
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        filled_shapes = cv2.cvtColor(filled_shapes, cv2.COLOR_GRAY2BGR)

        # add text on images
        cv2.putText(original, 'Original Image',
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)
        cv2.putText(gray, 'Grayscale Image',
                    self.position, FONT, CAPTION_SCALE,
                    BLACK, CAPTION_THICKNESS)
        cv2.putText(filled_shapes, 'Image after Thresholding',
                    self.position, FONT, CAPTION_SCALE,
                    WHITE, CAPTION_THICKNESS)

        # merge the 4 images together in a 2x2 matrix
        image = np.concatenate(
            (np.concatenate(
                (imutils.resize(original, self.width//2),
                 imutils.resize(gray, self.width//2)), axis=1),
             np.concatenate(
                (imutils.resize(filled_shapes, self.width//2),
                 imutils.resize(objects, self.width//2)), axis=1)), axis=0)

        cv2.imshow("Contouring Procedure", image)
        cv2.waitKey(0)


def main():

    image = Image(cv2.imread("shapes.jpg"))

    choice = 1
    while (choice):

        while(True):
            print("\n***Choose from following operations***\n")
            print("1.Show Image\n2.Rotation\n3.Smoothening")
            print("4.Convert to Grayscale\n5.Detect Edges\n6.Thresholding")
            print("7.Draw Contours/Count Objects\n0.Exit")

            try:
                choice = int(input("Enter choice: "))
                break
            except Exception:
                print("Please enter valid choice!!\n")

        if not choice:
            break
        if choice == 1:
            image.show(title="Shapes")
            continue
        if choice == 2:
            image.show(image.rotate(), "Rotated")
            continue
        if choice == 3:
            image.show(image.smoothen(), "Smoother Image")
            continue
        if choice == 4:
            image.show(image.grayscale(), "Grayscale Image")
            continue
        if choice == 5:
            image.show(image.detect_edges(), "Edges detected")
            continue
        if choice == 6:
            image.show(image.fill_shapes(
                image.grayscale()), "Filled shapes")
            continue
        if choice == 7:
            print("\nSteps for contour detection:")
            print("1. Color to Grayscale conversion")
            print("2. Thresholding to fill the shapes")
            print("3. Detecting, Counting and Highlighting Objects (contours)")
            gray_image = image.grayscale()
            filled_shapes = image.fill_shapes(gray_image)
            objects = image.draw_contours(filled_shapes)
            # show steps 1-3 in a single image
            image.show_contours(gray_image, filled_shapes, objects)
            continue


if __name__ == "__main__":
    main()
