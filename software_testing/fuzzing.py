import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt


class Fuzzing:
    def __init__(self):
        self.name = 'fuzzy test.'

    def add_cloud(self, imgs):
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


if __name__ == "__main__":
    # demo fuzzing
    filepath = '../data/leftImg8bit_trainvaltest/leftImg8bit/val/munster/munster_000000_000019_leftImg8bit.png'
    img = cv2.imread(filepath)
    fuzzier = Fuzzing()
    img = fuzzier.add_snowlandscape([img])
    plt.imshow(img[0])
    plt.show()
