import logging
from .logger import write_eval_log
from .evaluation import zero_shot_eval, linear_probe_eval


def get_val_sets(args):
    val_sets = []
    if args.ptbxl_path:
        val_sets.append("ptbxl_super_class")
        val_sets.append("ptbxl_sub_class")
        val_sets.append("ptbxl_form")
        val_sets.append("ptbxl_rhythm")
        
    if args.cpsc2018_path:
        val_sets.append("cpsc2018")

    return val_sets


def zero_shot_evaluatation(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}

    model.eval()

    val_sets = get_val_sets(args)

    for val_set in val_sets:
        eval_train, eval_test = data[f"train_{val_set}"].dataloader, data[f"val_{val_set}"].dataloader
        eval_model = model
        zero_shot_metrics = zero_shot_eval(eval_model, eval_test, args,
                                           tokenizer=tokenizer,
                                           dataset=val_set)
        metrics.update(
            {**zero_shot_metrics}
        )

    if not metrics:
        return metrics
    logging.info(
        f"Zero-shot eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"Zero-shot/" + name: val for name, val in metrics.items()}
    write_eval_log(args, log_data, data, epoch, metrics, tb_writer)
    return metrics


def linear_probing_evaluation(model, data, epoch, args, tb_writer=None):
    metrics = {}
    model.eval()

    val_sets = get_val_sets(args)

    for val_set in val_sets:
        eval_train, eval_test = data[f"train_{val_set}"].dataloader, data[f"val_{val_set}"].dataloader
        eval_model = model
        linear_probe_metrics = linear_probe_eval(eval_model, eval_train, eval_test, args, dataset=val_set)
        metrics.update(
            {**linear_probe_metrics}
        )

    if not metrics:
        return metrics

    logging.info(
        f"Linear-probe eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"Linear-probe/" + name: val for name, val in metrics.items()}
    write_eval_log(args, log_data, data, epoch, metrics, tb_writer)
    return metrics
