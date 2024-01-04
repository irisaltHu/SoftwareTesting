import os
import numpy as np
from mmengine.structures import PixelData
from metrics import Metrics, compute_difference
from fuzzer import Fuzzer
import cv2


class DataSelector:
    def __init__(self,
                 data_path,
                 label_path,
                 out_path,
                 top_k,
                 mutation_map=None,
                 use_increment=False
                 ):
        if mutation_map is None:
            mutation_map = {0: 'original', 1: 'fog', 2: 'rain', 3: 'snowlandscape', 4: 'cloud', 5: 'sunny',
                            6: 'gamma0.5', 7: 'gamma2.0'}
        self.data_path = data_path
        self.label_path = label_path
        self.out_path = out_path
        self.top_k = top_k  # only choose the most useful k mutations
        self.mutation_map = mutation_map
        self.use_increment = use_increment

    def select(self, models):
        """
        This works as a test executor and output analyzer
        Args:
            models: Model type which contains multiple or single model

        Returns:
            null, save the chosen mutations directly

        """
        folder_names = os.listdir(self.data_path)
        for folder_name in folder_names:
            file_path = self.data_path + folder_name + '/'
            label_file_path = self.label_path + folder_name + '/'
            filenames = os.listdir(file_path)

            for i in range(0, len(filenames)):
                if not filenames[i].endswith('8bit.png'):
                    continue

                print("selecting data of " + filenames[i])
                votes = [0 for i in range(len(self.mutation_map) - 1)]

                label_paths = label_file_path + filenames[i][:-15] + 'gtFine_labelIds.png'
                img_paths = [filenames[i], filenames[i][:-4] + '_fog.png', filenames[i][:-4] + '_rain.png',
                             filenames[i][:-4] + '_snowlandscape.png', filenames[i][:-4] + '_cloud.png',
                             filenames[i][:-4] + '_sunny.png', filenames[i][:-4] + '_gamma0.5.png',
                             filenames[i][:-4] + '_gamma2.0.png']
                img_paths = [file_path + img_path for img_path in img_paths]

                models_results = models.inference(img_paths)

                gt = models.load_label([label_paths])[0]

                # count difference in every model
                for model_results in models_results:
                    model_results[0].gt_sem_seg = PixelData(data=gt)

                    original_metric = Metrics()
                    original_metric.dataset_meta = dict(
                        classes=[item for item in models.classes],
                        label_map=dict(),
                        reduce_zero_label=False
                    )
                    original_metric.process([0] * 1, [model_results[0].to_dict()])
                    original_iou = original_metric.compute_iou(original_metric.results)
                    # print('-------------------------original-------------------------')
                    # print(original_iou)

                    # print('-------------------------mutation-------------------------')
                    differences = []
                    for j in range(1, len(model_results)):
                        model_results[j].gt_sem_seg = PixelData(data=gt[0])

                        metric = Metrics()
                        metric.dataset_meta = dict(
                            classes=[item for item in models.classes],
                            label_map=dict(),
                            reduce_zero_label=False
                        )
                        metric.process([0] * 1, [model_results[j].to_dict()])
                        mutation_iou = metric.compute_iou(metric.results)
                        # print(mutation_iou)
                        difference = compute_difference(original_iou, mutation_iou)

                        if self.use_increment:
                            differences.append(
                                np.nanmean([value for value in difference['increment'] if np.isfinite(value)])
                            )
                        else:
                            differences.append(
                                np.nanmean(
                                    [value for value in difference['increment_percentage'] if np.isfinite(value)])
                            )

                    # model vote for mutations
                    ranks = np.argsort(differences)[:self.top_k]
                    votes = [votes[i] + 1 if i in ranks else votes[i] for i in range(len(votes))]

                # save the selected mutations
                print('saving selected images')
                original_img = cv2.imread(img_paths[0])
                path = self.out_path + folder_name + '/' + filenames[i]
                cv2.imwrite(path, original_img)

                votes = np.argsort(votes)[:self.top_k] + 1

                cnt = 0
                fuzzer = Fuzzer()
                for j in range(self.top_k):
                    selected_img = cv2.imread(img_paths[votes[j]])

                    save_path = self.out_path + folder_name + '/' + \
                                filenames[i][:-4] + '_' + \
                                'mutation' + str(cnt) + '.png'
                    cv2.imwrite(save_path, selected_img)
                    cnt += 1

                    # multiple mutation
                    for k in range(j + 1, self.top_k):
                        if votes[k] == 1:
                            multi_mutation_img = fuzzer.add_fog([selected_img])[0]
                        elif votes[k] == 2:
                            multi_mutation_img = fuzzer.add_rain([selected_img])[0]
                        elif votes[k] == 3:
                            multi_mutation_img = fuzzer.add_snowlandscape([selected_img])[0]
                        elif votes[k] == 4:
                            multi_mutation_img = fuzzer.add_cloud([selected_img])[0]
                        elif votes[k] == 5:
                            multi_mutation_img = fuzzer.add_brightness([selected_img], 1.5)[0]
                        elif votes[k] == 6:
                            multi_mutation_img = fuzzer.gamma_transformation([selected_img], 0.5)[0]
                        elif votes[k] == 7:
                            multi_mutation_img = fuzzer.gamma_transformation([selected_img], 2.0)[0]

                        save_path = self.out_path + folder_name + '/' + \
                                    filenames[i][:-4] + '_' + \
                                    'mutation' + str(cnt) + '.png'
                        cv2.imwrite(save_path, multi_mutation_img)
                        cnt += 1

