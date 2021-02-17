from abc import abstractmethod
from typing import Any, Dict, List, Union, cast

import determined as det
from determined import horovod
from determined import lightning as dl
from determined_common import check

ValidateMetrics = Union[Dict, List]


class LightningTrialController(det.LoopTrialController):
    def __init__(self, trial_inst: det.Trial, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        check.is_instance(
            trial_inst, dl.LightningTrial, "LightningTrialController needs an LightningTrial"
        )
        self.trial = cast(dl.LightningTrial, trial_inst)
        self.context = cast(dl.LightningTrialContext, self.context)

        check.is_not_none(
            self.context.trainer,
            "Must call context.init_trainer in the LightningTrial.__init__",
        )
        self.context._validate_func = self.trial.validate

    @staticmethod
    def pre_execute_hook(env: det.EnvContext, hvd_config: horovod.HorovodContext) -> None:
        pass

    @staticmethod
    def from_trial(*args: Any, **kwargs: Any) -> det.TrialController:
        return LightningTrialController(*args, **kwargs)

    @staticmethod
    def from_native(*args: Any, **kwargs: Any) -> det.TrialController:
        raise NotImplementedError("LightningTrial only supports the Trial API")

    def run(self) -> None:
        try:
            self.trial.train()
        except det.errors.WorkerFinishedGracefully:
            pass


class LightningTrial(det.Trial):
    trial_controller_class = LightningTrialController
    trial_context_class = dl.LightningTrialContext

    @abstractmethod
    def __init__(self, context: dl.LightningTrialContext) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def validate(self) -> ValidateMetrics:
        pass
