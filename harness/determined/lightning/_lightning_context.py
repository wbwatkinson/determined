import logging
import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

import pytorch_lightning as ptl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler

import determined as det
from determined import errors, horovod, util, workload
from determined.layers import WorkloadCallbackHelper
from determined_common import check

VERY_LARGE_NUMBER = 9999999999999999
CHECKPOINT_FILE_NAME = "model.ckpt"


class LightningTrialContext(det.TrialContext):
    def __init__(
        self,
        env: det.EnvContext,
        workloads: workload.Stream,
        load_path: Optional[pathlib.Path],
        rendezvous_info: det.RendezvousInfo,
        hvd_config: horovod.HorovodContext,
    ):
        super().__init__(env, workloads, load_path, rendezvous_info, hvd_config)

        self.trainer = None  # type: Optional[ptl.Trainer]
        self._validate_func = None  # type: Optional[Callable]

    def init_trainer(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        callbacks: Optional[List[Callback]] = None,
        gradient_clip_val: float = 0,
        process_position: int = 0,
        log_gpu_memory: Optional[str] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: bool = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = "top",
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        profiler: Optional[Union[BaseProfiler, bool, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[list] = None,
        amp_backend: str = "native",
        amp_level: str = "O2",
        distributed_backend: Optional[str] = None,
        automatic_optimization: bool = True,
    ) -> None:
        # Merge callbacks.
        if not callbacks:
            callbacks = []
        callbacks.append(WorkloadCallback(self))

        # Load checkpoint. If there is checkpoint to resume from,
        # then override the user-specified checkpoint.
        if self.load_path:
            resume_from_checkpoint = self._load_path()

        # Merge arguments that control the max length.
        max_length = self.get_experiment_config()["searcher"]["max_length"]
        if "records" in max_length:
            max_epochs = VERY_LARGE_NUMBER
            max_steps = max_length["records"] // self.get_global_batch_size()
        elif "batches" in max_length:
            max_epochs = VERY_LARGE_NUMBER
            max_steps = max_length["batches"]
        elif "epochs" in max_length:
            max_epochs = max_length["epochs"]
            max_steps = None
        else:
            raise errors.InvalidConfigurationException(
                self.get_experiment_config(),
                "Experiment configuration must have searcher.max_length field",
            )

        # Set up Pytorch data parallel.
        # TODO: multiple GPU training
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/cluster.html

        # TODO: merge logger
        # logger =

        self.trainer = ptl.Trainer(
            logger=logger,
            checkpoint_callback=False,
            callbacks=callbacks,
            default_root_dir=os.getcwd(),
            gradient_clip_val=gradient_clip_val,
            process_position=process_position,
            num_nodes=self.distributed.get_num_agents(),
            num_processes=len(self.env.container_gpus),
            auto_select_gpus=True,
            tpu_cores=None,
            log_gpu_memory=log_gpu_memory,
            progress_bar_refresh_rate=0,
            overfit_batches=overfit_batches,
            track_grad_norm=track_grad_norm,
            check_val_every_n_epoch=check_val_every_n_epoch,
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            min_epochs=max_epochs,
            max_steps=max_steps,
            min_steps=max_steps,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            val_check_interval=val_check_interval,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            weights_summary=weights_summary,
            num_sanity_val_steps=num_sanity_val_steps,
            truncated_bptt_steps=truncated_bptt_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            reload_dataloaders_every_epoch=reload_dataloaders_every_epoch,
            auto_lr_find=auto_lr_find,
            replace_sampler_ddp=replace_sampler_ddp,
            terminate_on_nan=terminate_on_nan,
            auto_scale_batch_size=auto_scale_batch_size,
            prepare_data_per_node=prepare_data_per_node,
            plugins=plugins,
            amp_backend=amp_backend,
            amp_level=amp_level,
            distributed_backend=distributed_backend,
            automatic_optimization=automatic_optimization,
        )

    def _load_path(self) -> str:
        return str(cast(pathlib.Path, self.load_path).joinpath(CHECKPOINT_FILE_NAME))


class WorkloadCallback(Callback, WorkloadCallbackHelper):
    def __init__(self, context: LightningTrialContext):
        WorkloadCallbackHelper.__init__(self, context)
        Callback.__init__(self)

        self.context = cast(LightningTrialContext, self.context)

    def on_train_start(self, trainer, pl_module):  # type: ignore
        self.start_train()

    def on_train_batch_start(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> Any:
        self.start_train_next_batch()

    def on_train_batch_end(
        self,
        trainer: ptl.Trainer,
        pl_module: ptl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> Any:
        self.end_train_cur_batch(self._reduce_metrics(outputs))

    def on_train_epoch_start(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule) -> Any:
        self.start_train_next_epoch()

    def _reduce_metrics(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Reduce the metrics if no training_step_end is defined.
        if isinstance(outputs, torch.Tensor):
            return {"loss": outputs.cpu().detach().numpy()}
        elif isinstance(outputs, dict):
            for name, metric in outputs.items():
                # Convert PyTorch metric values to NumPy, so that
                # `det.util.encode_json` handles them properly without
                # needing a dependency on PyTorch.
                if isinstance(metric, torch.Tensor):
                    metric = metric.cpu().detach().numpy()
                outputs[name] = metric
            return outputs
        elif isinstance(outputs, list):
            all_opts_outs = {}
            for opt_idx, opt_all_gpu_outs in enumerate(outputs):
                check.is_instance(opt_all_gpu_outs, list)

                # Reduce across processes.
                opt_outs = util.make_step_metrics(None, opt_all_gpu_outs)["avg_metrics"]
                opt_outs = {f"opt_{opt_idx}_{k}": opt_outs[k] for k in opt_outs if opt_outs[k]}

                all_opts_outs.update(opt_outs)
            return all_opts_outs
        else:
            raise errors.InvalidMetricsTypeException

    def _reduce_validation_metrics(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Reduce the metrics if no training_step_end is defined.
        if isinstance(outputs, torch.Tensor):
            return {"loss": outputs.cpu().detach().numpy()}
        elif isinstance(outputs, dict):
            for name, metric in outputs.items():
                # Convert PyTorch metric values to NumPy, so that
                # `det.util.encode_json` handles them properly without
                # needing a dependency on PyTorch.
                if isinstance(metric, torch.Tensor):
                    metric = metric.cpu().detach().numpy()
                outputs[name] = metric
            return outputs
        # elif isinstance(outputs, list):
        #     all_opts_outs = {}
        #     for opt_idx, opt_all_gpu_outs in enumerate(outputs):
        #         check.is_instance(opt_all_gpu_outs, list)
        #
        #         # Reduce across processes.
        #         opt_outs = util.make_step_metrics(None, opt_all_gpu_outs)["avg_metrics"]
        #         opt_outs = {f"opt_{opt_idx}_{k}": opt_outs[k] for k in opt_outs if opt_outs[k]}
        #
        #         all_opts_outs.update(opt_outs)
        #     return all_opts_outs
        else:
            raise errors.InvalidMetricsTypeException

    def _compute_validation_metrics(self) -> workload.Response:
        self.context = cast(LightningTrialContext, self.context)
        self.context._validate_func = cast(Callable, self.context._validate_func)
        outputs = self.context._validate_func()
        return {
            "num_inputs": num_inputs,
            "validation_metrics": self._reduce_validation_metrics(outputs),
        }

    def _save(self, path: pathlib.Path) -> workload.Response:
        self.context = cast(LightningTrialContext, self.context)
        self.context.trainer = cast(ptl.Trainer, self.context.trainer)
        self.context.trainer.save_checkpoint(path.joinpath(CHECKPOINT_FILE_NAME))
        return {
            "framework": f"pytorch-lightning-{ptl.__version__}",
            "format": "",
        }
