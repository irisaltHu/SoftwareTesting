import torch
from mmseg.apis import init_model, inference_model
import os
import cv2
from mmengine.structures import PixelData
from mmseg.datasets import cityscapes
import metrics


class Models:
    def __init__(self,
                 data_path,
                 label_path,
                 class_map=None,
                 label_map=cityscapes.CityscapesDataset.METAINFO,
                 ):
        if class_map is None:
            class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                         26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.data_path = data_path
        self.label_path = label_path
        self.class_map = class_map
        self.label_map = label_map
        self.classes = label_map['classes']

        self.models = []

    def append_model(self, config_path, checkpoint_path):
        model = init_model(config_path, checkpoint_path)
        self.models.append(model)

    def inference(self, img_paths):
        ret = []
        for model in self.models:
            ret.append(inference_model(model, img_paths))
        return ret

    def traverse(self, batch_size=4):
        folder_names = os.listdir(self.data_path)

        for folder_name in folder_names:
            file_path = self.data_path + folder_name
            filenames = os.listdir(file_path)

            for i in range(0, int(len(filenames) / batch_size), batch_size):
                img_paths = filenames[i * batch_size:(i + 1) * batch_size]
                label_prefix = self.label_path + folder_name
                label_paths = [label_prefix + '/' + img[:-15] + 'gtFine_labelIds.png' for img in img_paths]
                img_paths = [file_path + '/' + img for img in img_paths]

                model_results = self.inference(img_paths)

                for model_result in model_results:
                    for j in range(batch_size):
                        label = cv2.imread(label_paths[j])
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
                        gt = [gt]
                        gt = torch.Tensor(gt)
                        model_result[j].gt_sem_seg = PixelData(data=gt)

                    metric = metrics.Metrics()
                    metric.dataset_meta = dict(
                        classes=[item for item in self.classes],
                        label_map=dict(),
                        reduce_zero_label=False
                    )
                    metric.process([0] * len(model_results), [result.to_dict() for result in model_result])
                    output = metric.compute_iou(metric.results)
                    output = metric.compute_difference(output, output - 1)

                    return output

