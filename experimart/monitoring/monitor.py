class PerformanceMonitor:
    def __init__(
        self,
        metrics: dict = None,
        trackers: dict = None,
    ):
        self._trackers = trackers
        self._metrics = metrics

    def __call__(self, epoch: int, outputs: dict):
        statistics = {"epoch": epoch}
        for mode, metrics in self._metrics.items():
            for metric_key, metric in metrics.items():
                if metric_key in outputs[mode]:
                    statistics[f"{mode}_{metric_key}"] = metric(outputs[mode][metric_key])

        for tracker in self._trackers.values():
            tracker.update(statistics)
