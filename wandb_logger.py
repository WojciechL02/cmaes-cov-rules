from typing import List, Optional
import wandb


class Logger:
    """Characterizes a Weights and Biases wandb logger
    Assumes prior wandb login (wandb login)"""

    def __init__(
            self,
            exp_name: Optional[str] = None,
    ):
        wandb.init(
            group=exp_name,
        )
        self.metrics = []

    def log_args(self, args):
        wandb.config.update(args)
    
    def log_scalar(self, iter, value, func, group):
        key = "{}/{}".format(group, func)

        iter_key = "_iter/" + key.replace("/", "_")
        if key not in self.metrics:
            wandb.define_metric(iter_key, hidden=True)
            wandb.define_metric(key, step_metric=iter_key)
            self.metrics.append(key)
        wandb.log({key: value, iter_key: iter})
    
    def log_ecdf(self, xs, ys, func):
        key = "ecdf/{}".format(func)

        iter_key = "_iter/" + key.replace("/", "_")
        if key not in self.metrics:
            wandb.define_metric(iter_key, hidden=True)
            wandb.define_metric(key, step_metric=iter_key)
            self.metrics.append(key)
        
        for x, y in zip(xs, ys):
            wandb.log({key: y, iter_key: x})
        
        wandb.log({
            f"custom_{key}": wandb.plot.line_series(
                xs=xs.tolist(),
                ys=[ys.tolist()],
                keys=["ecdf"],
                title=f"{key}",
                xname="log10(fitness)"
            )
        })

    def __del__(self):
        wandb.finish()
