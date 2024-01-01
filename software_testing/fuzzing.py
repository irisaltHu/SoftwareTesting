import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Fuzzer:
    def add_cloud(self, imgs):
        """
        This works for every method in class Fuzzer
        Args:
            imgs: a list of images(ndarray)

        Returns:
            a list of the mutations(ndarray)
        """
        aug = iaa.Clouds()
        return aug.augment(images=imgs)

    def add_fog(self, imgs):
        aug = iaa.Fog()
        return aug.augment(images=imgs)

    # add rain drop
    def add_rain(self, imgs):
        aug = iaa.Rain()
        return aug.augment(images=imgs)

    # add snow on the landscape
    def add_snowlandscape(self, imgs):
        aug = iaa.FastSnowyLandscape()
        return aug.augment(images=imgs)

    # add a layer of rain(not covering the whole img)
    def add_rainlayer(self, imgs):
        aug = iaa.RainLayer(drop_size=(0.1, 0.4), speed=(0.01, 0.05),
                            density=0.5, density_uniformity=0.5,
                            drop_size_uniformity=0.5, angle=10,
                            blur_sigma_fraction=0.0005)
        return aug.augment(images=imgs)

    # add a layer of snow(not covering the whole img)
    def add_snowlayer(self, imgs):
        aug = iaa.SnowflakesLayer(flake_size=(0.1, 0.4), speed=(0.01, 0.05),
                                  density=0.5, density_uniformity=0.5,
                                  flake_size_uniformity=0.5, angle=10,
                                  blur_sigma_fraction=0.0005)
        return aug.augment(images=imgs)

    # gamma transformation
    def gamma_transformation(self, imgs, gamma):
        ret = []
        for img in imgs:
            img -= img.min()
            img = img / (img.max() - img.min())
            img = np.power(img, gamma) * 255
            img = np.uint8(img)
            ret.append(img)
        return ret

    def add_brightness(self, imgs, brightness_coefficient):
        ret = []
        for img in imgs:
            img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            img_hls = np.array(img_hls, dtype=np.float64)
            img_hls[:, :, 1] = img_hls[:, :, 1] * brightness_coefficient
            img_hls[:, :, 1][img_hls[:, :, 1] > 255] = 255
            img_hls = np.array(img_hls, dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)
            ret.append(img_rgb)
        return ret
