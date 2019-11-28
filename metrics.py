from catalyst.dl.core import MetricCallback
from catalyst.dl.utils import criterion
import torch.nn.functional as F


def mae(input, target):
    return F.l1_loss(input, target, reduction='mean')


class MAEMetric(MetricCallback):
    """
    Dice metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "mae",
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=mae,
            input_key=input_key,
            output_key=output_key,
        )
