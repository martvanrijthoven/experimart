from experiment.step import StepIterator
import torch


def get_model_parameters(model):
    return model.parameters()


def save_model(model, output_folder, suffix):
    torch.save(model.state_dict(), output_folder / f"{suffix}.pth")


class TorchTrainingStepIterator(StepIterator):
    def __init__(
        self, model, data_iterator, num_steps, optimizer, criterion, scheduler
    ):
        super().__init__(model, data_iterator, num_steps)
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler

    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label, info = next(self._data_iterator)
            self._optimizer.zero_grad()
            output = self._model(data)["out"]
            loss = self._criterion(output, label)
            loss.backward()
            self._optimizer.step()
            yield loss.item()
        self._scheduler.step()


class TorchValidationStepIterator(StepIterator):
    def __init__(self, model, data_iterator, num_steps, criterion):
        super().__init__(model, data_iterator, num_steps)
        self._criterion = criterion

    def steps(self, data, label):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label, info = next(self._data_iterator)
                output = self._model(data)
                output = output["out"]
                loss = self._criterion(output, label)
                yield loss.item()
