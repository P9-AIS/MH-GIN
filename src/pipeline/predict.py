import torch
from tqdm import tqdm
from src.utils.utils import log_metrics

def predict(model, test_loader, logger, coordinate_is_mae_smape=False, mean_test=False):
    with torch.no_grad():
        model.eval()
        eval_metrics = {}
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                (loss, loss_list), eval_dict = model(test_batch, evaluate=True)
                for key, value in eval_dict.items():
                    if key not in eval_metrics:
                        eval_metrics[key] = 0
                    eval_metrics[key] += value

        for key in eval_metrics:
            eval_metrics[key] /= len(test_loader)
        log_metrics(logger, eval_metrics, coordinate_is_mae_smape=coordinate_is_mae_smape, mean_test=mean_test)
        return eval_metrics
