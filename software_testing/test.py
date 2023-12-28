import numpy as np
import torch
from mmseg.apis import init_model, inference_model
import matplotlib.pyplot as plt
import os
import cv2
from mmengine.structures import PixelData
from mmseg.datasets import cityscapes
from mmseg.evaluation.metrics import iou_metric
import metrics


class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
             26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
label_map = cityscapes.CityscapesDataset.METAINFO
classes = label_map['classes']


def main():
    # load model
    config_path = '../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_path = 'pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth'
    model = init_model(config_path, checkpoint_path)

    # path of val set
    data_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/val/'
    folder_names = os.listdir(data_path)

    # traverse all the folder in val set
    for folder_name in folder_names:
        file_path = data_path + folder_name
        filenames = os.listdir(file_path)
        batch_size = 1

        # test using batch
        for i in range(0, int(len(filenames) / batch_size), batch_size):
            # paths
            img_paths = filenames[i * batch_size:(i + 1) * batch_size]
            label_prefix = '../data/gtFine_trainvaltest/gtFine/val/' + folder_name
            label_paths = [label_prefix + '/' + img[:-15] + 'gtFine_labelIds.png' for img in img_paths]
            img_paths = [file_path + '/' + img for img in img_paths]

            # inference
            results = inference_model(model, img_paths)

            # load ground truth
            for j in range(batch_size):
                label = cv2.imread(label_paths[j])
                height = label.shape[0]
                width = label.shape[1]
                gt = []
                for n in range(height):
                    gt_line = []
                    for m in range(width):
                        index = class_map.get(label[n][m].tolist()[0], 19)
                        if index == 19:
                            gt_line.append(int(255))
                        else:
                            gt_line.append(index)
                    gt.append(gt_line)
                gt = [gt]
                gt = torch.Tensor(gt)
                results[j].gt_sem_seg = PixelData(data=gt)

            metric = metrics.Metrics()
            metric.dataset_meta = dict(
                    classes=[item for item in classes],
                    label_map=dict(),
                    reduce_zero_label=False
                )
            metric.process([0] * len(results), [result.to_dict() for result in results])
            output = metric.compute_iou(metric.results)
            output = metric.compute_difference(output, output - 1)

            # metric from mmseg
            # metric = iou_metric.IoUMetric()
            # metric.dataset_meta = dict(
            #         classes=[item for item in classes],
            #         label_map=dict(),
            #         reduce_zero_label=False
            #     )

            # compute IoU and Acc
            # metric.process([0] * len(results), [result.to_dict() for result in results])
            # output = metric.compute_metrics(metric.results)

            # visualization of the inference results
            # for result in results:
            #     img = result.pred_sem_seg.get('data').cpu()
            #     img = img.numpy()[0]
            #     plt.imshow(img)
            #     plt.show()


if __name__ == "__main__":
    main()
