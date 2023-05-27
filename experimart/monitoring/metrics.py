from pandas import DataFrame

from experimart.monitoring.tracking import Tracker


class MetricSummary:
    """
    epoch callback
    dict: list -> dict: value
    e.g.,
        dict: list[floats] -> dict: value (e.g, mean(loss))
        dict: list[object] -> dict: value (e.g., confusion matrix)

    """

    def __init__(
        self,
        tracker: Tracker = None,
        metrics: dict = None,
    ):
        self._tracker = tracker
        self._metrics = metrics

    def __call__(self, epoch: int, outputs: dict):
        statistics = {"epoch": epoch}
        for mode, metrics in self._metrics.items():
            for metric_key, metric in metrics.items():
                statistics[f"{mode}_{metric_key}"] = metric(outputs[mode][metric_key])

        if self._tracker is not None:
            self._tracker.update(statistics)
