from fuzzing import Fuzzer
import numpy as np
import os
import cv2


class DataGenerator:
    def __init__(self,
                 data_path,
                 out_path,
                 sample_per_folder):
        self.data_path = data_path
        self.out_path = out_path
        self.sample_per_folder = sample_per_folder
        self.fuzzer = Fuzzer()

    def generate(self):

        folder_names = os.listdir(self.data_path)
        for folder_name in folder_names:
            # os.mkdir(self.out_path + folder_name)

            file_path = self.data_path + folder_name
            filenames = os.listdir(file_path)

            indices = np.random.randint(0, len(filenames), 30)
            target_filenames = [file_path + '/' + filenames[index] for index in indices]

            for i in range(len(target_filenames)):
                # carry out mutations
                print('generate mutaion data of ' + target_filenames[i])
                img = cv2.imread(target_filenames[i])
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]], img)

                img_fog = self.fuzzer.add_fog([img])[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_fog.png', img_fog)

                img_rain = self.fuzzer.add_rain([img])[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_rain.png', img_rain)

                img_snowlandscape = self.fuzzer.add_snowlandscape([img])[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_snowlandscape.png',
                            img_snowlandscape)

                img_cloud = self.fuzzer.add_cloud([img])[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_cloud.png', img_cloud)

                img_sunny = self.fuzzer.add_brightness([img], 1.5)[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_sunny.png', img_sunny)

                img_gamma_half = self.fuzzer.gamma_transformation([img], 0.5)[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_gamma0.5.png', img_gamma_half)

                img_gamma_2 = self.fuzzer.gamma_transformation([img], 2.0)[0]
                cv2.imwrite(self.out_path + folder_name + '/' + filenames[indices[i]][:-4] + '_gamma2.0.png', img_gamma_2)
