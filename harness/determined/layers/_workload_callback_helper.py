import logging
import pathlib
from abc import abstractmethod
from typing import Any, Dict, List, cast

import determined as det
from determined import errors, util, workload
from determined_common import check


class WorkloadCallbackHelper:
    """
    WorkloadCallbackHelper helps any model training loops interact with the Determined
    cluster. It fetches and executes workloads.

    Each framework or library can implement a callback that inherits from this class
    and run its own training loop while responding to the workloads from the master.
    """

    def __init__(self, context: det.TrialContext):
        self.context = context
        self._workloads = context.workloads
        self._workload_iter = iter(self._workloads)

        self.step_metrics = []  # type: List[Dict]

    def fetch_next_workload(self) -> None:
        self._cur_workload, self._cur_args, self._cur_response_func = next(self._workload_iter)

    def run_cur_workload(self) -> None:
        """
        This function executes the current workload and fetches the next workload
        until it hits a RUN_STEP workload.
        """
        while True:
            w, args, response_func = self._cur_workload, self._cur_args, self._cur_response_func

            if w.kind == workload.Workload.Kind.RUN_STEP:
                cur_total_batches = self.context.cur_total_batch_idx + 1
                check.true(
                    cur_total_batches <= w.total_batches()
                    and cur_total_batches >= w.total_batches_processed,
                    f"Current total batches should be within the step: "
                    f"[w.total_batches_processed, {w.total_batches()}]",
                )
                if cur_total_batches == w.total_batches():
                    try:
                        if self.context.is_chief:
                            check.is_instance(self.step_metrics, List)
                            metrics = cast(
                                workload.Response,
                                util.make_step_metrics(w.num_batches, self.step_metrics),
                            )
                        else:
                            metrics = workload.Skipped()

                        response_func(
                            util.wrap_metrics(
                                metrics,
                                self.context.get_stop_requested(),
                                invalid_hp=False,
                            )
                        )
                    except det.InvalidHP as e:
                        logging.info(
                            "Invalid hyperparameter exception in trial train step: {}".format(e)
                        )
                        response_func(
                            util.wrap_metrics(
                                {},
                                self.context.get_stop_requested(),
                                invalid_hp=True,
                            )
                        )
                    self.reset_step_metrics()
                else:
                    break

            elif w.kind == workload.Workload.Kind.COMPUTE_VALIDATION_METRICS:
                try:
                    metrics = self._compute_validation_metrics()
                    if self.context.is_chief:
                        check.is_instance(
                            metrics, dict, "must return dictionary for validation metrics"
                        )
                        util.check_val_metrics(cast(Dict[str, Any], metrics))
                    else:
                        metrics = workload.Skipped()

                    response_func(
                        util.wrap_metrics(
                            metrics,
                            self.context.get_stop_requested(),
                            invalid_hp=False,
                        )
                    )
                except det.InvalidHP as e:
                    logging.info(
                        "Invalid hyperparameter exception in trial validation step: {}".format(e)
                    )
                    response_func(
                        util.wrap_metrics(
                            {},
                            self.context.get_stop_requested(),
                            invalid_hp=True,
                        )
                    )

            elif w.kind == workload.Workload.Kind.CHECKPOINT_MODEL:
                check.eq(len(args), 1)
                check.is_instance(args[0], pathlib.Path)
                path = cast(pathlib.Path, args[0])
                response_func(self._save(path))

            elif w.kind == workload.Workload.Kind.TERMINATE:
                response_func({} if self.context.is_chief else workload.Skipped())
                raise errors.WorkerFinishedGracefully("Exiting normally.")

            else:
                raise AssertionError("Unexpected workload: {}".format(w.kind))

            self.fetch_next_workload()

    def start_train(self) -> None:
        # TODO: fast-forward self.context.cur_batch_idx to the checkpoint
        self.context.cur_epoch_idx = -1
        self.context.cur_total_batch_idx = -1
        self.fetch_next_workload()
        self.run_cur_workload()

    def start_train_next_epoch(self) -> None:
        self.context.cur_epoch_idx += 1

    def start_train_next_batch(self) -> None:
        self.context.cur_total_batch_idx += 1

    def end_train_cur_batch(self, metrics: Dict[str, Any]) -> None:
        self.add_train_batch_metrics(metrics)
        self.run_cur_workload()

    def add_train_batch_metrics(self, metrics: Any) -> None:
        check.is_instance(
            metrics,
            dict,
            "reduced metrics must be a dictionary "
            f"mapping string names to Tensor metrics, got {type(metrics)}",
        )
        self.step_metrics.append(metrics)

    def reset_step_metrics(self) -> None:
        self.step_metrics = []

    @abstractmethod
    def _compute_validation_metrics(self) -> workload.Response:
        pass

    @abstractmethod
    def _save(self, path: pathlib.Path) -> workload.Response:
        pass
