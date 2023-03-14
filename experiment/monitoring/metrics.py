from experiment.monitoring.tracking import Tracker

class MetricSummary:
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
        
        print (metrics)
        if self._tracker is not None:
            self._tracker.update(metrics)
