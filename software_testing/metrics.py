import numpy as np
from collections import OrderedDict
from typing import Dict
from mmseg.evaluation.metrics.iou_metric import IoUMetric


def compute_difference(original_iou, new_iou):
    increment = new_iou - original_iou
    increment_percentage = increment / original_iou
    return {'increment': increment, 'increment_percentage': increment_percentage}


class Metrics(IoUMetric):
    def __init__(self, **kwargs):
        super(Metrics, self).__init__()

    def compute_iou(self, results: list) -> Dict[str, float]:
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        return ret_metrics_class['IoU']

