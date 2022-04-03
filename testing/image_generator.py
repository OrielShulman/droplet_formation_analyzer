import copy
import numpy as np
import random
import cv2
from typing import Tuple



def add_salt_And_pepper(base_image: np.ndarray) -> np.ndarray:
    # 0 * 255 = 0 !
    # noise = np.random.choice([0, 255], size=base_image.shape, p=[0.99, 0.01])
    # return base_image * noise

    salted = copy.deepcopy(base_image)
    for i in range(int(np.sqrt(base_image.size))):
        b = random.randint(0, base_image.size - 1)
        salted[np.unravel_index(b, base_image.shape)] = 0

        w = random.randint(0, base_image.size - 1)
        salted[np.unravel_index(w, base_image.shape)] = 255

    return salted


def add_drops(base_image: np.ndarray) -> np.ndarray:
    rows, cols = base_image.shape

    drop_img = copy.deepcopy(base_image)

    for row in range(5, rows, 20):
        for col in range(5, cols, 20):
            if col + 4 < cols and row + 4 < rows:
                for i in range(5):
                    for j in range(5):
                        drop_img[row + i, col + j] = random.randint(200, 255)
                # drop_img[row:row+5, col:col+5] = random.randint(150, 255)

    return drop_img


def create_raw_droplet_image(size=(1000, 1000), with_noise: bool = False):
    image = np.zeros(size)
    image = add_drops(image)
    if with_noise:
        image = add_salt_And_pepper(image)
    return image


def get_testing_sample() -> Tuple[np.ndarray, np.ndarray]:
    dots_image = cv2.imread('resources/dots.jpg', 0)
    dot = cv2.imread('resources/dot.jpg', 0)
    return dots_image, dot


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    # im_size = 100
    # black_image = np.zeros((im_size, im_size))
    # # dr.display_image(black_image)
    # t1 = time.time()
    # dotted = add_drops(base_image=black_image)
    # print(f"time: {time.time() - t1}\n")
    # # dr.display_image(dotted)
    # t1 = time.time()
    # noise_image = add_salt_And_pepper(base_image=dotted)
    # print(f"time: {time.time() - t1}\n")
    # dr.display_image(noise_image)
    #
    # dr.display_image(create_raw_droplet_image(size=(100, 100), with_noise=True))



    # noise = np.random.choice([0, 255], size=(1000, 1000), p=[0.99, 0.01])
    # dr.display_image(noise)


    # example = np.array([[1, 2, 3], [4, 5, 6]])
    # print(example)
    #
    # print(np.unravel_index(5, (3, 3)))
    #
    # example[np.unravel_index(5, example.shape)] = 20
    #
    # print(example)





