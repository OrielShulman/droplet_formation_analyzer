import numpy as np
import imageio as img
from skimage.color import rgb2gray
import os
from typing import Optional
# import skimage
import matplotlib.pyplot as plt


class Display:

    G_SCALE = 0
    RGB_SCALE = 1

    def __init__(self, filename: Optional[str] = None):

        self.image: Optional[np.ndarray] = None
        if filename is not None:
            self.read_image(filename=filename)

    def read_image(self, filename) -> None:
        """
        this function reads a grayscale image file.
        if an RGB image was given, converts it into a grayscale representation.
        @:param filename: the filename of an image, could be grayscale or RGB.
        @:return: None. apply the image to self.image. represented by a matrix of type np.float64 in range [0, 255].
        """

        image = img.imread(filename)  # maybe divide by the max index (255)

        if len(image.shape) > 2:
            image = rgb2gray(image)

        self.image = np.array(image).astype(np.float64)

    def display_image(self):
        if self.image is None:
            raise ValueError("No image was loaded")
        else:
            plt.imshow(self.image, cmap="gray")
            plt.axis('off')

            plt.show()


def display_image(image: np.ndarray) -> None:
    """
    displays an image grayscale or RGB
    :param image: the image to display
    """
    if len(image.shape) > 2:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap="gray")

    plt.axis('off')
    plt.show()


def read_grayscale_image(image_filename: str) -> np.ndarray:
    """
    reads an image and returns it in grayscale representation as an np.ndarray of type float64
    :param image_filename: the filename of an image, could be grayscale or RGB.
    :return: a grayscale image
    """
    image = img.imread(image_filename)
    if len(image.shape) > 2:
        image = rgb2gray(image)
    return np.array(image).astype(np.float64)


def read_RGB_image(image_filename: str) -> np.ndarray:
    """
    reads an image and returns it in grayscale representation as an np.ndarray of type float64
    :param image_filename: the filename of an image, could be grayscale or RGB.
    :return: a grayscale image
    """
    image = img.imread(image_filename)
    return np.array(image).astype(np.float64)


if __name__ == '__main__':
    cactus_img = Display('resources/cactus.jpg')
    dots_img = Display('resources/dots.jpg')
    cactus_img.display_image()
    dots_img.display_image()

    # a = read_grayscale_image('resources/cactus.jpg')
    # b = read_grayscale_image('resources/dots.jpg')
    # display_image(a)
    # display_image(b)

    # a = read_RGB_image('resources/cactus.jpg')
    # b = read_RGB_image('resources/dots.jpg')
    # display_image(a)
    # display_image(b)


