from fuzzing import Fuzzer
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def main():
    fuzzer = Fuzzer()
    data_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/val/'
    out_path = '../data/AI_test/'

    folder_names = os.listdir(data_path)
    for folder_name in folder_names:
        # os.mkdir(out_path + folder_name)

        file_path = data_path + folder_name
        filenames = os.listdir(file_path)

        indices = np.random.randint(0, len(filenames), 30)
        target_filenames = [file_path + '/' + filenames[index] for index in indices]

        for i in range(len(target_filenames)):
            print(target_filenames[i])
            img = cv2.imread(target_filenames[i])
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]], img)

            img_fog = fuzzer.add_fog([img])[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_fog.png', img_fog)

            img_rain = fuzzer.add_rain([img])[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_rain.png', img_rain)

            img_snowlandscape = fuzzer.add_snowlandscape([img])[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_snowlandscape.png', img_snowlandscape)

            img_cloud = fuzzer.add_cloud([img])[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_cloud.png', img_cloud)

            img_sunny = fuzzer.add_brightness([img], 1.5)[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_sunny.png', img_sunny)

            img_gamma_half = fuzzer.gamma_transformation([img], 0.5)[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_gamma0.5.png', img_gamma_half)

            img_gamma_2 = fuzzer.gamma_transformation([img], 2.0)[0]
            cv2.imwrite(out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_gamma2.0.png', img_gamma_2)


if __name__ == "__main__":
    main()
