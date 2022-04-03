import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict
import cv2
from skimage.feature import peak_local_max


class ImageAnalyzer:
    DROPLET_SIZE = 7  # size in pixels

    def __init__(self, image: np.ndarray, droplet_template: np.ndarray = None):
        self.image: np.ndarray = image  # a grayscale image an 2D numpy array
        self.template: np.ndarray = droplet_template

        self.droplets_num: Optional[int] = None  # number of droplets fount in image
        self.droplets: Dict[tuple, float]  # a dict of: {coordinate: intensity}
        self.droplets_average_intensity: Optional[float] = None

    def find_droplets(self) -> int:
        """
        analyzing the sample image, find the droplets and classify them by intensity
        :return:
        """
        # find matching dots:
        res = cv2.matchTemplate(self.image, self.template, cv2.TM_CCOEFF_NORMED)  # minimum square difference
        res[res < 0.5] = 0
        # res[res < np.mean(res)] = 0
        # dr.display_image(res)

        # find peaks coordinates:
        offset = self.template.shape[0] // 2  # due to edges loss when cropping
        coordinates = peak_local_max(res, min_distance=self.DROPLET_SIZE) + offset
        # TODO: for each coordinate: save intensity in original image

        # plot resault - delete later
        plt.imshow(self.image, cmap=plt.cm.gray)
        x, y = zip(*coordinates[::-1])
        plt.scatter(y, x, s=self.DROPLET_SIZE * 3, facecolors='none', edgecolors='r')

        # for dot in coordinates

        #     cv2.circle(self.image, dot, self.DROPLET_SIZE, (255, 0, 0), thickness=2)
        # cv2.imshow("OK", self.image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        plt.axis('off')
        plt.show()

        return coordinates.shape[0]

    # # minimum square difference
    # res = cv2.matchTemplate(self.image, self.template, cv2.TM_SQDIFF)
    # # res[res < 0.7] = 0
    # # res[res < np.mean(res)] = 0
    # dr.display_image(res)
    #
    # # minimum square difference
    # res = cv2.matchTemplate(self.image, self.template, cv2.TM_SQDIFF_NORMED)
    # # res[res < np.mean(res)] = 0
    # # res[res < 0.7] = 0
    # dr.display_image(res)

    # finds also where size is biger
    # res = cv2.matchTemplate(self.image, self.template, cv2.TM_CCOEFF)
    # dr.display_image(res)
    # print(res.max())
    # # res[res < np.mean(res)] = 0
    # res[res < 0.7] = 0
    # dr.display_image(res)

    def plot_droplets(self) -> None:
        # TODO: can keep droplet by multiplying the peak img with the original one after threashold applying
        pass

    def plot_droplets_intensity(self) -> None:
        """
        plots the droplets as a heatmap of intensities over the original image
        """
        pass


if __name__ == '__main__':
    import testing.image_generator as ig

    # droplet = np.zeros((4, 4))
    # droplet[1:3, 1:3] = 1
    # print(droplet.shape)
    # dr.display_image(droplet)
    # base_image = ig.create_raw_droplet_image(size=(300, 400), with_noise=True)
    # dr.display_image(base_image)

    base_image, droplet = ig.get_testing_sample()

    analyzer = ImageAnalyzer(image=base_image, droplet_template=droplet)

    print(analyzer.find_droplets())
