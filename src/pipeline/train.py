import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from src.utils.utils import log_metrics


def scalarize(x):
    """Ensure tensor is reduced to a scalar for logging and backward."""
    if torch.is_tensor(x):
        return x.mean() if x.numel() > 1 else x
    return x


def train(model, args, logger, train_loader, valid_loader=None, folder_name=""):
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=float(args.weight_decay))
    is_lr_decay = args.use_lr_schedule
    if is_lr_decay:
        p1 = int(0.75 * args.epochs)
        p2 = int(0.9 * args.epochs)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )

    valid_epoch_interval = args.valid_epoch_interval
    best_valid_loss = float("inf")
    best_train_loss = float("inf")
    no_improvement_count = 0
    patience = args.patience
    start_epoch = 0

    if args.resume:
        if not os.path.exists(args.resume):
            logger.warning(f"Checkpoint file {args.resume} not found.")
            exit()
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if is_lr_decay and checkpoint["scheduler_state_dict"] is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_valid_loss = checkpoint.get("best_valid_loss", best_valid_loss)
        best_train_loss = checkpoint.get("best_train_loss", best_train_loss)
        no_improvement_count = checkpoint.get("no_improvement_count", 0)
        start_epoch = checkpoint["epoch"] + 1
        logger.info(
            f"Resumed training from {args.resume} at epoch {start_epoch}")
    else:
        logger.info("Starting training from scratch")

    if folder_name != "":
        output_path = folder_name + "/model.pth"

    for epoch_no in range(start_epoch, args.epochs):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                optimizer.zero_grad()
                (loss, loss_list), _ = model(batch, True)

                # Reduce losses to scalars
                loss = scalarize(loss)
                loss_list = [scalarize(l) for l in loss_list]

                loss.backward()

                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (
                        torch.isnan(param.grad).any() or torch.isinf(
                            param.grad).any()
                    ):
                        has_nan_grad = True
                        logger.warning(f"NaN/Inf gradient detected in {name}")
                        param.grad = torch.zeros_like(param.grad)

                if has_nan_grad:
                    logger.warning(
                        f"Skipping update due to NaN/Inf gradients at epoch {epoch_no}, batch {batch_no}"
                    )
                else:
                    avg_loss += loss.item()
                    optimizer.step()

                logger.info(
                    f"Epoch {epoch_no}: train_loss:{loss.item():.4f}, "
                    f"spatial:{loss_list[0].item():.4f}, "
                    f"temporal:{loss_list[1].item():.4f}, "
                    f"angular:{loss_list[2].item():.4f}, "
                    f"continuous:{loss_list[3].item():.4f}, "
                    f"discrete:{loss_list[4].item():.4f}"
                )

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            avg_train_loss = avg_loss / len(train_loader)
            logger.info(
                f"Epoch {epoch_no}: avg_train_loss = {avg_train_loss:.4f}")

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                logger.info(
                    f"Epoch {epoch_no}: Best training loss updated to {best_train_loss:.4f}"
                )
                if folder_name != "":
                    checkpoint = {
                        "epoch": epoch_no,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": (
                            lr_scheduler.state_dict() if is_lr_decay else None
                        ),
                        "best_valid_loss": best_valid_loss,
                        "best_train_loss": best_train_loss,
                        "no_improvement_count": no_improvement_count,
                    }
                    os.makedirs(folder_name, exist_ok=True)
                    torch.save(checkpoint, output_path)

            if is_lr_decay:
                lr_scheduler.step()

            if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
                model.eval()
                avg_loss_valid_list = [0, 0, 0, 0, 0, 0]
                eval_metrics = {}
                with torch.no_grad():
                    with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                        for batch_no, valid_batch in enumerate(it, start=1):
                            (loss, loss_list), eval_dict = model(
                                valid_batch, True
                            )
                            loss = scalarize(loss)
                            loss_list = [scalarize(l) for l in loss_list]

                            avg_loss_valid_list[0] += loss_list[0].item()
                            avg_loss_valid_list[1] += loss_list[1].item()
                            avg_loss_valid_list[2] += loss_list[2].item()
                            avg_loss_valid_list[3] += loss_list[3].item()
                            avg_loss_valid_list[4] += loss_list[4].item()
                            avg_loss_valid_list[5] += sum(l.item()
                                                          for l in loss_list)

                            for key, value in eval_dict.items():
                                eval_metrics[key] = eval_metrics.get(
                                    key, 0) + value

                            it.set_postfix(
                                ordered_dict={
                                    "valid_avg_epoch_loss": avg_loss_valid_list[5]
                                    / batch_no,
                                    "epoch": epoch_no,
                                },
                                refresh=False,
                            )

                        avg_valid_loss = avg_loss_valid_list[5] / \
                            len(valid_loader)
                        for key in eval_metrics:
                            eval_metrics[key] /= len(valid_loader)
                        log_message = (
                            f"Epoch {epoch_no}: avg_valid_loss = {avg_valid_loss:.4f}, "
                            f"Spatial: {avg_loss_valid_list[0] / len(valid_loader):.4f}, "
                            f"Temporal: {avg_loss_valid_list[1] / len(valid_loader):.4f}, "
                            f"Angular: {avg_loss_valid_list[2] / len(valid_loader):.4f}, "
                            f"Continuous: {avg_loss_valid_list[3] / len(valid_loader):.4f}, "
                            f"Discrete: {avg_loss_valid_list[4] / len(valid_loader):.4f}"
                        )
                        logger.info(log_message)

                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    no_improvement_count = 0
                    print(
                        f"Epoch {epoch_no}: Best validation loss updated to {best_valid_loss:.4f}"
                    )
                    logger.info(
                        f"Epoch {epoch_no}: Best validation loss updated to {best_valid_loss:.4f}"
                    )
                    if folder_name != "":
                        checkpoint = {
                            "epoch": epoch_no,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": (
                                lr_scheduler.state_dict() if is_lr_decay else None
                            ),
                            "best_valid_loss": best_valid_loss,
                            "best_train_loss": best_train_loss,
                            "no_improvement_count": no_improvement_count,
                        }
                        os.makedirs(folder_name, exist_ok=True)
                        torch.save(
                            checkpoint, folder_name +
                            f"/tmp_model{epoch_no}.pth"
                        )
                    log_metrics(logger, eval_metrics,
                                coordinate_is_mae_smape=False)
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch_no} "
                            f"after {patience} epochs without improvement."
                        )
                        break
