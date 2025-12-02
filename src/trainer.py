#necessary imports for training
import argparse
import os
import random
import time
from typing import Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from preprocessing import preprocess
from distance_matrix import DIST_MATRIX
from wandb_help import (build_config, init_wandb, log_model_info, log_epoch_metrics, finish_wandb, log_distance_epoch_metrics, log_distance_histograms,)

#wandb project name
WANDB_PROJECT = "UCF_COMPUTER_VISION"

#sets random seeds for reproducibility across runs
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#builds normalized distance penalty matrix for loss computation
def build_distance_penalty_matrix(
    lambda_penalty: float,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    #converts distance matrix to np array
    dm_np = np.asarray(DIST_MATRIX, dtype=np.float32)
    #validates matrix is square
    if dm_np.shape[0] != dm_np.shape[1]:
        raise ValueError(f"DIST_MATRIX must be square, got {dm_np.shape}")
    #validates matrix size matches classes
    if dm_np.shape[0] != num_classes:
        raise ValueError(
            f"DIST_MATRIX size {dm_np.shape[0]} does not match num_classes {num_classes}"
        )
    #converts to torch tensor for GPU computation
    dm = torch.from_numpy(dm_np)

    #inverse-distance mode (λ < 0), penalizes close predictions more
    if lambda_penalty < 0.0:
        #sets basic matrix to zero ready for inversion
        inv = torch.zeros_like(dm)
        #makes a mask for wherever dm is not 0 for the inversion and avoids div by 0
        mask = dm > 0
        inv[mask] = 1.0 / dm[mask]

        #takes max inverse element
        max_inv = inv.max()
        #if the maximum element is negative or 0 no need to do any altering
        if max_inv <= 0:
            dm_norm = inv.clone()
        else:
            #does the normalization
            dm_norm = inv / max_inv

        #lambda used is absolute value at end of day, just used to invert it
        lambda_eff = float(-lambda_penalty)
    else:
        #standard mode, penalizes far predictions more
        #takes max
        max_val = dm.max()
        #checks to see if positive
        if max_val <= 0:
            #if not then no need to adjust
            dm_norm = dm.clone()
        else:
            #normalize
            dm_norm = dm / max_val
        #lambda scalar, e.g. effective lambda is just the lambda already used
        lambda_eff = float(lambda_penalty)

    #moves to device, returns effective lambda
    return dm_norm.to(device), lambda_eff

#computes loss with distance penalty term
def compute_loss_with_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_criterion: nn.Module,
    distance_matrix: Optional[torch.Tensor],
    lambda_eff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #computes cross entropy loss
    ce_loss = ce_criterion(logits, targets)

    #if no distance penalty (λ = 0), return cross entropy alone, additionally if its significantly small assume 0.
    if distance_matrix is None or abs(lambda_eff) < 1e-12:
        #penalty is set to a bunch of 0's and we just return cronss entropy loss as only relevant value since penalty is all zeros.
        penalty = torch.zeros(1, device=logits.device)
        return ce_loss, ce_loss.detach(), penalty.detach()
    #computes softmax probabilities over classes
    probs = torch.softmax(logits, dim=1)
    #selects distance row for each true label y
    D_y = distance_matrix[targets] 
    #computes expected distance
    sample_penalty = (probs * D_y).sum(dim=1)
    #means over varying elements for scalar
    penalty = sample_penalty.mean()
    #total loss = CE + λ * distance_penalty
    total_loss = ce_loss + lambda_eff * penalty
    #returns loss, cross entropy and penalty scores alone as well
    return total_loss, ce_loss.detach(), penalty.detach()

#formats filename with hyperparameters
def format_checkpoint_name(model_name: str, lam: float, k: int, lr: float) -> str:
    lam_str = str(lam)
    lr_str = str(lr)
    #checkpoint name below
    return f"{model_name}_{lam_str}_k{k}_{lr_str}.pt"

#approximates FLOPS, does not give true calculation
def estimate_epoch_flops(trainable_params: int, num_samples: int) -> float:
    #assumes 4 FLOPS per paramater per sample
    return float(4.0 * trainable_params * num_samples)

#trains model for one epoch
def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, ce_criterion: nn.Module, distance_matrix: Optional[torch.Tensor], lambda_eff: float, device: torch.device,) -> Tuple[float, float, float]:
    #sets model to training mode
    model.train()
    #loss, cross entropy, penalty, total correct, and total metrics init to 0
    total_loss = 0.0
    total_ce = 0.0
    total_pen = 0.0
    correct = 0
    total = 0

    #iterates through the batch
    for images, labels in train_loader:
        #moves batch to device (train and labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        #zeros gradients
        optimizer.zero_grad(set_to_none=True)
        #forward pass
        logits = model(images)
        #computes loss (CE + distance penalty)
        loss, ce_loss, pen_loss = compute_loss_with_penalty(
            logits, labels, ce_criterion, distance_matrix, lambda_eff
        )
        #backward pass
        loss.backward()
        #update model parameters
        optimizer.step()
        
        #adds to metrics, total adds batch size as count, adds to total loss, cross entropy, and penalty.
        batch_size = labels.size(0)
        total += batch_size
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_pen += pen_loss.item() * batch_size

        #computes batch accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    #computes epoch averages over all metrics
    avg_loss = total_loss / total
    avg_ce = total_ce / total
    avg_pen = total_pen / total
    acc = correct / total
    #returns loss, penalty, and accuracy, cross entropy alone not really required
    return avg_loss, acc, avg_pen

#evaluates model on test set with distance metrics
def test_one_epoch(model: nn.Module, test_loader: DataLoader, ce_criterion: nn.Module, distance_matrix: Optional[torch.Tensor], lambda_eff: float, device: torch.device,) -> Tuple[float, float, float, Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray],]:
    #sets model to evaluation mode
    model.eval()
    #more metrics, such as loss, cross entropy, penalty, etc.
    total_loss = 0.0
    total_ce = 0.0
    total_pen = 0.0
    correct = 0
    total = 0

    #overall distance stats, which sums distance and count errors
    sum_dist_all = 0.0
    sum_dist_err = 0.0
    count_all = 0
    count_err = 0

    #per-class distance counters, similar fashion as above
    class_sum_all = None
    class_count_all = None
    class_sum_err = None
    class_count_err = None

    #checks for distance matrix
    if distance_matrix is not None:
        #the number of classes is size of distance matrix as it is initialized where each column/row represents one class
        num_classes = distance_matrix.size(0)
        #initializes per-class counters
        class_sum_all = np.zeros(num_classes, dtype=np.float64)
        class_count_all = np.zeros(num_classes, dtype=np.int64)
        class_sum_err = np.zeros(num_classes, dtype=np.float64)
        class_count_err = np.zeros(num_classes, dtype=np.int64)
    #since eval, we turn off grad collection
    with torch.no_grad():
        #iterates through batch in test loader
        for images, labels in test_loader:
            #moves batch to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            #forward pass only, gets final output
            logits = model(images)
            #collects loss
            loss, ce_loss, pen_loss = compute_loss_with_penalty(
                logits, labels, ce_criterion, distance_matrix, lambda_eff
            )

            #collects loss metrics based on total count and scaling by loss details
            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_pen += pen_loss.item() * batch_size

            #computes batch accuracy, by collecting predictions and then comparing to labels
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

            #checks to see for existence of distance matrix
            if distance_matrix is not None:
                #collects distance based on labels and predictions
                batch_dists = distance_matrix[labels, preds]
                #for each element in the batch just add the corresponding distance
                sum_dist_all += batch_dists.sum().item()
                #adds total count by the batch size
                count_all += batch_size

                #accumulates distances for errors only, does not care about correct classifications
                err_mask = preds != labels
                #checks to see if any errors, if so then we only add the distance and count for misaligned labels and predictions
                if err_mask.any():
                    #collects distances
                    err_dists = batch_dists[err_mask]
                    #adds distance errors
                    sum_dist_err += err_dists.sum().item()
                    #adds to the count of errors
                    count_err += err_dists.numel()

                #converts per-sample distance penalties to CPU NumPy for easy processing with labels, predictions, and distances
                labels_cpu = labels.detach().to("cpu").numpy()
                preds_cpu = preds.detach().to("cpu").numpy()
                dists_cpu = batch_dists.detach().to("cpu").numpy()
                #loops through each sample in the batch, but the np variant
                for y, d, y_pred in zip(labels_cpu, dists_cpu, preds_cpu):
                    #all samples sum
                    class_sum_all[y] += float(d)
                    class_count_all[y] += 1
                    #errors only sum
                    if y_pred != y:
                        class_sum_err[y] += float(d)
                        class_count_err[y] += 1

    #computes epoch averages
    avg_loss = total_loss / total
    avg_ce = total_ce / total
    avg_pen = total_pen / total
    acc = correct / total

    #computes mean distances over all samples including the errors
    mean_dist_all = None
    mean_dist_errors = None
    if distance_matrix is not None and count_all > 0:
        mean_dist_all = sum_dist_all / count_all
    if distance_matrix is not None and count_err > 0:
        mean_dist_errors = sum_dist_err / count_err

    #computes per-class mean distances, sets up mean of all and error count distances
    class_mean_dist_all = None
    class_mean_dist_errors = None
    #checks to see if distance matrix exists and sum all exists already
    if distance_matrix is not None and class_sum_all is not None:
        #per-class means including correct predictions
        class_mean_dist_all = np.zeros_like(class_sum_all, dtype=np.float32)
        nonzero_all = class_count_all > 0
        class_mean_dist_all[nonzero_all] = (class_sum_all[nonzero_all] / class_count_all[nonzero_all]).astype(np.float32)

        #per-class means over errors only, ignores samples where predictions are correct.
        class_mean_dist_errors = np.zeros_like(class_sum_err, dtype=np.float32)
        nonzero_err = class_count_err > 0
        class_mean_dist_errors[nonzero_err] = (
            class_sum_err[nonzero_err] / class_count_err[nonzero_err]
        ).astype(np.float32)
    #returns performance (loss, acc, distance) metrics
    return (avg_loss, acc, avg_pen, mean_dist_all, mean_dist_errors, class_mean_dist_all, class_mean_dist_errors,)

#parses arguments
def parse_args():
    #inits parser
    parser = argparse.ArgumentParser(description="ViT trainer with distance-aware loss")

    #model architecture
    parser.add_argument(
        "--model",
        type=str,
        default="vit_tiny_patch16_224",
    )
    #k-dataset
    parser.add_argument(
        "--k",
        type=int,
        default=1,
    )
    #preprocessing type: 'base' or 'distort'
    parser.add_argument(
        "--prep_type",
        type=str,
        default="base",
    )
    #batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    #num of epochs
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    #learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    #weight decay
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    #number of workers used when prep of data
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    #training device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    #seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    #directory for checkpoints
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
    )
    #distance penalty weight lambda
    parser.add_argument(
        "--lambda_penalty",
        type=float,
        default=0.0,
    )
    #returns args effectively
    return parser.parse_args()


def main():
    #parses arguments
    args = parse_args()
    set_seed(args.seed)

    #sets compute device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #opens a directory for output
    os.makedirs(args.output_dir, exist_ok=True)

    #logs the config to user to ensure correctness
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    #preprocesses data with chosen k-split and prep type
    print("\nPreprocessing data...")
    bundle = preprocess(k=args.k, prep_type=args.prep_type)

    #sets the appropriate datasets and labels as necessary
    train_ds = bundle["train"]
    test_ds = bundle["test"]
    labels = bundle.get("labels", None)
    num_classes = len(labels) if labels is not None else len(set(train_ds.labels))

    print(f"Num classes: {num_classes}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    #creates dataloaders
    train_loader = DataLoader( train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,)

    #builds pretrained ViT model with custom head
    model = timm.create_model(
        args.model,
        pretrained=True,  #uses ImageNet pretrained, we are fine tuning and pretraining ourselves
        num_classes=num_classes,  #adapts final layer to our classes
    )
    #sets model to device
    model.to(device)

    #creates cross-entropy loss
    ce_criterion = nn.CrossEntropyLoss()

    #builds distance matrix
    dist_matrix, lambda_eff = build_distance_penalty_matrix(
        args.lambda_penalty, num_classes, device
    )
    #checks to see if lambda is signficant enough by |lambda| < 10^-12.
    if abs(args.lambda_penalty) < 1e-12:
        lambda_mode = "none"
    elif args.lambda_penalty > 0:
        lambda_mode = "distance"
    else:
        lambda_mode = "inverse_distance"

    #creates AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    #formats checkpoint filename with hyperparameters
    base_ckpt_name = format_checkpoint_name(args.model, args.lambda_penalty, args.k, args.lr)

    #builds wandb run name
    run_name = f"{args.model}_k{args.k}_lam{args.lambda_penalty}_lr{args.lr}_{args.prep_type}"

    #builds wandb config
    cfg = build_config(
        model_name=args.model,
        k=args.k,
        prep_type=args.prep_type,
        lambda_reg=args.lambda_penalty,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        extra={
            "lambda_eff": lambda_eff,
            "lambda_mode": lambda_mode,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "num_workers": args.num_workers,
            "output_dir": args.output_dir,
            "num_classes": num_classes,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
        },
    )

    #initializes wandb run
    init_wandb(
        project_name=WANDB_PROJECT,
        run_name=run_name,
        config=cfg,
    )

    #logs model info to wandb
    trainable_params = log_model_info(
        model,
        summary_extra={
            "num_classes": num_classes,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
        },
    )

    #estimates computational cost per epoch
    approx_epoch_flops = estimate_epoch_flops(trainable_params, len(train_ds))

    #tracks best model for checkpointing, and saving best pth based on highest test acc
    best_test_acc = 0.0
    best_ckpt_path = None
    best_epoch = None
    best_class_mean_dist_all = None
    best_class_mean_dist_errors = None

    #training and test loop simultanouesly
    for epoch in range(1, args.epochs + 1):
        print(f"\n Epoch {epoch}/{args.epochs}")
        epoch_start_time = time.time()

        #trains model for one epoch
        train_loss, train_acc, train_pen = train_one_epoch(model, train_loader, optimizer, ce_criterion, dist_matrix, lambda_eff, device,)
        print(
            f"Train | loss: {train_loss:.4f} "
            f"(penalty: {train_pen:.4f}) | acc: {train_acc*100:.2f}%"
        )

        #evaluates on test set
        (test_loss, test_acc, test_pen, test_mean_dist_all, test_mean_dist_errors, test_class_mean_all, test_class_mean_errors,) = test_one_epoch(model, test_loader, ce_criterion, dist_matrix, lambda_eff, device,)
        print(
            f"Test | loss: {test_loss:.4f} "
            f"(penalty: {test_pen:.4f}) | acc: {test_acc*100:.2f}%"
        )
        if test_mean_dist_all is not None:
            print(f"mean_dist_all: {test_mean_dist_all:.4f}")
        if test_mean_dist_errors is not None:
            print(f"mean_dist_errors: {test_mean_dist_errors:.4f}")
        
        #collects time by current minus start
        epoch_time_sec = time.time() - epoch_start_time

        #logs standard metrics to wandb
        log_epoch_metrics(epoch=epoch, train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc, epoch_time_sec=epoch_time_sec, epoch_flops=approx_epoch_flops,)

        #logs distance-specific metrics to wandb
        if dist_matrix is not None:
            log_distance_epoch_metrics(epoch=epoch, split="test", mean_dist_all=test_mean_dist_all, mean_dist_errors=test_mean_dist_errors,)

        #saves checkpoint if new best test acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            ckpt_name = base_ckpt_name
            best_ckpt_path = os.path.join(args.output_dir, ckpt_name)
            #saves model state, epoch, and hyperparameters to output
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

            #saves per-class distance stats from best epoch, logged at very end of run, when training loop over
            if test_class_mean_all is not None:
                best_class_mean_dist_all = test_class_mean_all.copy()
            if test_class_mean_errors is not None:
                best_class_mean_dist_errors = test_class_mean_errors.copy()

    #logs distance histograms
    if dist_matrix is not None and best_class_mean_dist_all is not None:
        log_distance_histograms(split="test", class_mean_dist_all=best_class_mean_dist_all, class_mean_dist_errors=best_class_mean_dist_errors, bins=None, step=best_epoch,)

    #finishes wandb run and finalizes logs
    finish_wandb()
    
    print("\nTraining finished.")
    #if best test exists we log which was the best epoch, what the highest test acc was, and what the best checkpoint was
    if best_ckpt_path is not None:
        print(f"Best epoch: {best_epoch}")
        print(f"Best test acc: {best_test_acc*100:.2f}%")
        print(f"Best checkpoint: {best_ckpt_path}")
    else:
        print("No checkpoint was saved.")


if __name__ == "__main__":
    main()