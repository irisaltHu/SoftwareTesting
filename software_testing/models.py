import numpy as np
import torch
from mmseg.apis import init_model, inference_model
import os
import cv2
from mmengine.structures import PixelData
from mmseg.datasets import cityscapes
from metrics import Metrics, compute_difference


class Models:
    def __init__(self,
                 data_path,
                 label_path,
                 out_path,
                 class_map=None,
                 label_map=cityscapes.CityscapesDataset.METAINFO,
                 ):
        # choose the main 19 classes of cityscapes dataset
        if class_map is None:
            class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                         26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.data_path = data_path
        self.label_path = label_path
        self.class_map = class_map
        self.label_map = label_map
        self.out_path = out_path
        self.classes = label_map['classes']

        self.models = []

    # add a new model
    def append_model(self, config_path, checkpoint_path):
        model = init_model(config_path, checkpoint_path)
        self.models.append(model)

    def inference(self, img_paths):
        """

        Args:
            img_paths: required to be a list of str

        Returns:
            a list of inference results of all the model
        """
        ret = []
        for model in self.models:
            results = []
            for img_path in img_paths:
                result = inference_model(model, img_path)
                results.append(result)

            # This is a quick method that requires much GPU memory
            # results = inference_model(model, img_paths)

            ret.append(results)
        return ret

    def load_label(self, label_paths):
        """

        Args:
            label_paths: a list of paths of target label images

        Returns:
            a list of ground truth

        """
        labels = []
        for i in range(len(label_paths)):
            label = cv2.imread(label_paths[i])
            height = label.shape[0]
            width = label.shape[1]
            gt = []
            for n in range(height):
                gt_line = []
                for m in range(width):
                    index = self.class_map.get(label[n][m].tolist()[0], 19)
                    if index == 19:
                        gt_line.append(int(255))
                    else:
                        gt_line.append(index)
                gt.append(gt_line)
            labels.append(torch.Tensor([gt]))
        return labels

    def traverse(self, top_k):
        """
        Usage:
            traverse the test dataset and print the metrics which judge the test dataset and model
        Args:
            top_k: number of mutations chosen in selection

        Returns:
            null, print the metrics directly

        """
        folder_names = os.listdir(self.data_path)

        for folder_name in folder_names:
            file_path = self.data_path + folder_name + '/'
            filenames = os.listdir(file_path)

            results0 = [[0 for _ in range(len(self.classes))] for _ in range(len(self.models))]
            results1 = [[0 for _ in range(len(self.classes))] for _ in range(len(self.models))]
            total = [[0 for _ in range(len(self.classes))] for _ in range(len(self.models))]

            for i in range(len(filenames)):
                if not filenames[i].endswith('8bit.png'):
                    continue
                print('-----------------------------------------------------------------------')
                print('test on ' + filenames[i] + ' and its mutations')
                img_paths = []
                for j in range(int(top_k * (top_k + 1) / 2)):
                    img_paths.append(filenames[i][:-4] + '_mutation' + str(j) + '.png')

                label_prefix = self.label_path + folder_name
                label_path = label_prefix + '/' + filenames[i][:-15] + 'gtFine_labelIds.png'
                label = self.load_label([label_path])[0]
                img_paths = [file_path + img for img in img_paths]

                # compute iou on original image
                original_results = self.inference([file_path + filenames[i]])
                for j in range(len(original_results)):
                    original_results[j][0].gt_sem_seg = PixelData(data=label)
                    original_metric = Metrics()
                    original_metric.dataset_meta = dict(
                        classes=[item for item in self.classes],
                        label_map=dict(),
                        reduce_zero_label=False
                    )
                    original_metric.process([0] * 1, [original_results[j][0].to_dict()])
                    original_IoU = original_metric.compute_iou(original_metric.results)

                models_results = self.inference(img_paths)

                for j in range(len(models_results)):
                    for k in range(int(top_k * (top_k + 1) / 2)):
                        models_results[j][k].gt_sem_seg = PixelData(data=label)

                    metric = Metrics()
                    metric.dataset_meta = dict(
                        classes=[item for item in self.classes],
                        label_map=dict(),
                        reduce_zero_label=False
                    )
                    metric.process([0] * top_k, [result.to_dict() for result in models_results[j]])
                    mutation_mIoU = metric.compute_iou(metric.results)
                    increment = compute_difference(original_IoU, mutation_mIoU)


                    # print("metrics on model" + str(j + 1) + ":")
                    # output = {
                    #     'increment': {
                    #         self.classes[x]: increment['increment'][x]
                    #         for x in range(len(increment['increment']))
                    #     },
                    #     'increment_percentage': {
                    #         self.classes[x]: increment['increment_percentage'][x] * 100
                    #         for x in range(len(increment['increment_percentage']))
                    #     }
                    # }
                    # print(output)

                    for x in range(len(increment['increment'])):
                        if np.isfinite(increment['increment'][x]) and np.isfinite(increment['increment_percentage'][x]):
                            results0[j][x] += increment['increment'][x]
                            results1[j][x] += increment['increment_percentage'][x]
                            total[j][x] += 1

            # average metrics in different city
            with open(self.out_path, 'a') as f:
                print("The avg metrics in folder " + folder_name + " is:")
                f.write("The avg metrics in folder " + folder_name + " is:\n")
                for i in range(len(self.models)):
                    f.write("model" + str(i) + ":\n")
                    avg_increment = [results0[i][x] / total[i][x] if total[i][x] != 0 else 0.0 for x in range(len(self.classes))]
                    avg_increment_percentage = [results1[i][x] / total[i][x] if total[i][x] != 0 else 0.0 for x in range(len(self.classes))]
                    output = {
                        'increment': {
                            self.classes[x]: avg_increment[x]
                            for x in range(len(avg_increment))
                        },
                        'increment_percentage': {
                            self.classes[x]: avg_increment_percentage[x] * 100
                            for x in range(len(avg_increment_percentage))
                        }
                    }
                    print(output)
                    f.write(str(output) + '\n')
        return
