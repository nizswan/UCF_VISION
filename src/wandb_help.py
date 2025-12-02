# helps log to wandb for general purposes, allows for construction of neat and nice graphs
from __future__ import annotations
import os
from typing import Any, Dict, Optional, Sequence
import numpy as np
#inits run to none as default
RUN = None

#builds a configuration
def build_config(model_name: str, k: int, prep_type: str, lambda_reg: float, batch_size: int, epochs: int, lr: float, extra: Optional[Dict[str, Any]] = None,) -> Dict[str, Any]:
    #sets up configuration type with model name, k, preparation type, lambda regularization constant, batch size, epoch count, learning rate, etc
    cfg: Dict[str, Any] = {
        "model_name": model_name,
        "k": k,
        "prep_type": prep_type,
        "lambda_reg": lambda_reg,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
    }
    #adds additional details in configuration that may or may not be necessary
    if extra:
        cfg.update(extra)
    #returns configuration
    return cfg
#initializes a wandb run
def init_wandb(
    project_name: str,
    run_name: str,
    config: Dict[str, Any],
) -> None:
    #collects project name (where I log to), run name (what project is called), and configuration
    global RUN
    #collects api key from env
    api_key = os.getenv("WANDB_API_KEY")
    #if no api key then error
    if not api_key:
        print("[wandb_help] WANDB_API_KEY not set; WandB logging is DISABLED for this run.")
        RUN = None
        return
    #imports wandb here, so the rest of the program can still work if wandb does not work
    import wandb

    # sets run, which was none above, to the following
    RUN = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
    )
    print(f"WandB run initialized to project='{project_name}' under run name='{run_name}'")

#logs model information given model and information
def log_model_info(
    model,
    summary_extra: Optional[Dict[str, Any]] = None,
) -> int:
    # counts trainable params only
    trainable_params = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))

    # only logs to wandb if a run is active
    if RUN is not None:
        import wandb

        # store parameter count in wandb summary
        wandb.run.summary["trainable_params"] = int(trainable_params)

        # optionally log any additional provided metadata (e.g., depth, classes, samples)
        if summary_extra:
            for k, v in summary_extra.items():
                wandb.run.summary[k] = v

    # return param count so trainer can print/use it
    return trainable_params

#sets up some of the graphs on wandb
def log_epoch_metrics(epoch: int, train_loss: Optional[float] = None, train_acc: Optional[float] = None, test_loss: Optional[float] = None, test_acc: Optional[float] = None, epoch_time_sec: Optional[float] = None, epoch_flops: Optional[float] = None,) -> None:

    # if wandb isn't active, do nothing
    if RUN is None:
        return
    
    # reimports since wandb is not global library
    import wandb

    # always track epoch index in logs
    log_dict: Dict[str, Any] = {"epoch": epoch}

    # sets train loss and train accuracy
    if train_loss is not None:
        log_dict["train/loss"] = float(train_loss)
    if train_acc is not None:
        log_dict["train/accuracy"] = float(train_acc)

    # sets test loss and test accuracy
    if test_loss is not None:
        log_dict["test/loss"] = float(test_loss)
    if test_acc is not None:
        log_dict["test/accuracy"] = float(test_acc)

    # sets up time per epoch and flop count per epoch
    if epoch_time_sec is not None:
        log_dict["epoch/time_sec"] = float(epoch_time_sec)
    if epoch_flops is not None:
        log_dict["epoch/flops"] = float(epoch_flops)
    
    # only log if we have something beyond epoch
    if len(log_dict) > 1:
        wandb.log(log_dict, step=epoch)


def log_scalar(name: str, value: float, step: Optional[int] = None) -> None:
    # if wandb is disabled, skip logging
    if RUN is None:
        return
    import wandb
    # if caller specifies a step index, bind the scalar to that step
    if step is not None:
        wandb.log({name: float(value)}, step=step)
    else:
        # otherwise let wandb decide the step automatically
        wandb.log({name: float(value)})

#logs epoch metrics
def log_distance_epoch_metrics(epoch: int, split: str, mean_dist_all: Optional[float] = None, mean_dist_errors: Optional[float] = None,) -> None:
    # skip if wandb logging isn't active
    if RUN is None:
        return
    import wandb
    # container for any distance metrics we actually have this epoch
    log_dict: Dict[str, Any] = {}
    # mean distance over all samples (correct = 0, errors > 0)
    if mean_dist_all is not None:
        log_dict[f"{split}/mean_dist_all"] = float(mean_dist_all)
    # mean distance over only misclassified samples
    if mean_dist_errors is not None:
        log_dict[f"{split}/mean_dist_errors"] = float(mean_dist_errors)
    # log only if we actually collected something
    if log_dict:
        wandb.log(log_dict, step=epoch)

#logs histograms
def log_distance_histograms(split: str, class_mean_dist_all: Optional[Sequence[float]] = None, class_mean_dist_errors: Optional[Sequence[float]] = None, bins: Optional[Sequence[float]] = None, step: Optional[int] = None,) -> None:
    # skip if wandb is not active
    if RUN is None:
        return
    import wandb
    # sets default bin edges if none provided
    if bins is None:
        # 5 bins: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bins = np.linspace(0.0, 1.0, 6, dtype=float)
    else:
        # convert custom bins to numpy array
        bins = np.asarray(bins, dtype=float)
    # container for histogram objects to log
    log_dict: Dict[str, Any] = {}
    # histogram over per-class mean distances (non errors included, e.g. predictions that were correct)
    if class_mean_dist_all is not None:
        data_all = np.asarray(class_mean_dist_all, dtype=float)
        # uses numpy histogram to build wandb histogram object
        hist_all = wandb.Histogram(np_histogram=np.histogram(data_all, bins=bins))
        log_dict[f"{split}/class_mean_dist_all_hist"] = hist_all
    # histogram over per-class mean distances (errors only, none that are correct)
    if class_mean_dist_errors is not None:
        data_err = np.asarray(class_mean_dist_errors, dtype=float)
        hist_err = wandb.Histogram(np_histogram=np.histogram(data_err, bins=bins))
        log_dict[f"{split}/class_mean_dist_errors_hist"] = hist_err
    # log if anything was populated
    if log_dict:
        wandb.log(log_dict, step=step)



def finish_wandb() -> None:
    global RUN
    # if no active wandb run, nothing to clean up
    if RUN is None:
        return
    
    import wandb
    # close the active wandb run
    wandb.run.finish()
    RUN = None  # reset global handle for future calls
    print("WandB run finished.")
