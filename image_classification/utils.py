import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def show_image(images, labels, num_images=4):
    image = make_grid(images[:num_images])
    image = image/2 + 0.5
    np_img = image.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()