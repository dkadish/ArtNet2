# These decorators helps with distributed settings
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced

from artnet.utils.coco_eval import CocoEvaluator


class CocoMetricBase(Metric):
    '''
    reset() is triggered every EPOCH_STARTED (See Events).

    update() is triggered every ITERATION_COMPLETED.

    compute() is triggered every EPOCH_COMPLETED.
    '''

    def __init__(self, coco_api_val_dataset, iou_types, output_transform=lambda x: x):
        self.coco_api_val_dataset = coco_api_val_dataset
        self.iou_types = iou_types
        # self.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types)

        super(CocoMetricBase, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):  # EPOCH_STARTED
        # self._num_correct = 0
        # self._num_examples = 0
        super(CocoMetricBase, self).reset()
        print('Metric IOU Types: {}'.format(self.iou_types))
        self.coco_evaluator = CocoEvaluator(self.coco_api_val_dataset, self.iou_types)  # Events.Started > Evaluator

    @reinit__is_reduced
    def update(self, output):  # ITERATION_COMPLETED
        # y_pred, y = output
        images, targets, res = output

        try:
            self.coco_evaluator.update(res)
        except TypeError as e:
            # The model/target produced no bounding boxes and cannot be evaluated for this iteration.
            pass

        # indices = torch.argmax(y_pred, dim=1)
        #
        # mask = (y != self.ignored_class)
        # mask &= (indices != self.ignored_class)
        # y = y[mask]
        # indices = indices[mask]
        # correct = torch.eq(indices, y).view(-1)
        #
        # self._num_correct += torch.sum(correct).item()
        # self._num_examples += correct.shape[0]

    # @sync_all_reduce("coco_evaluator")
    def compute(self):  # EPOCH_COMPLETED
        # if self._num_examples == 0:
        #     raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        # return self._num_correct / self._num_examples

        self.coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        # From Events.EPOCH_COMPLETED > trainer
        for res_type in self.coco_evaluator.iou_types:
            ap, ap5, ap75 = self.coco_evaluator.coco_eval[res_type].stats[:3]
            # writer.add_scalar("validation-{}/average precision 0_5".format(res_type), average_precision_05,
            #                   engine.state.iteration)

        return ap, ap5, ap75


class CocoAP(CocoMetricBase):
    def compute(self):
        ap, ap5, ap75 = super().compute()
        return ap


class CocoAP5(CocoMetricBase):
    def compute(self):
        ap, ap5, ap75 = super().compute()
        return ap5


class CocoAP75(CocoMetricBase):
    def compute(self):
        ap, ap5, ap75 = super().compute()
        return ap75