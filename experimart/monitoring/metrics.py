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
        self, train_callables: dict, validation_callables: dict, tracker: Tracker = None,
    ):
        self._tracker = tracker
        self._callables = {"train": train_callables, "validation": validation_callables}

    def __call__(self, epoch, train_output, validation_output):
        outputs = {"train": train_output, "validation": validation_output}
        metrics = {"epoch": epoch}
        for name, callables in self._callables.items():
            for key, callable in callables.items():
                metrics[f"{name}_{key}"] = callable(outputs[name][key])
        
        print(metrics)
        if self._tracker is not None:
            self._tracker.update(metrics)
