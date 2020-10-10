from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class CocoMetricBase(Metric):
    '''
    reset() is triggered every EPOCH_STARTED (See Events).

    update() is triggered every ITERATION_COMPLETED.

    compute() is triggered every EPOCH_COMPLETED.
    '''

    def __init__(self, ignored_class, output_transform=lambda x: x):
        self.ignored_class = ignored_class
        self._num_correct = None
        self._num_examples = None
        super(CocoMetricBase, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self): # EPOCH_STARTED
        self._num_correct = 0
        self._num_examples = 0
        super(CocoMetricBase, self).reset()

        engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types) #Events.Started > Evaluator

    @reinit__is_reduced
    def update(self, output): # ITERATION_COMPLETED
        y_pred, y = output

        indices = torch.argmax(y_pred, dim=1)

        mask = (y != self.ignored_class)
        mask &= (indices != self.ignored_class)
        y = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self): # EPOCH_COMPLETED
        # if self._num_examples == 0:
        #     raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        # return self._num_correct / self._num_examples

        # From Events.EPOCH_COMPLETED > trainer
        for res_type in evaluator.state.coco_evaluator.iou_types:
            average_precision_05 = evaluator.state.coco_evaluator.coco_eval[res_type].stats[1]
            # writer.add_scalar("validation-{}/average precision 0_5".format(res_type), average_precision_05,
            #                   engine.state.iteration)

        return average_precision_05