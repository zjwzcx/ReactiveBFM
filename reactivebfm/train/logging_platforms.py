import os
import glob
import datetime


class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_scalars(self, values, iteration, group_name=None):
        for name, value in values.items():
            self.report_scalar(name, value, iteration, group_name)

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='RL',
                              task_name=name)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir, *args, **kwargs):
        pass

class WandBPlatform(TrainPlatform):
    import wandb
    DEFAULT_PROJECT = os.getenv("WANDB_PROJECT", "reactivebfm")

    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)

        self.wandb.login(
            host=os.getenv("WANDB_BASE_URL"),
            key=os.getenv("WANDB_API_KEY"),
        )
        time_str = self.get_time_str()
        exp_name = f"{time_str}_{self.get_run_base_name()}"
        init_kwargs = dict(
            project=self.DEFAULT_PROJECT,
            name=exp_name,
            group=self.get_run_base_name(),
            id=exp_name,  # in order to send continued runs to the same record
            # resume='allow',  # in order to send continued runs to the same record
            reinit=True,
            save_code=True,
            config=config,  # config can also be sent via report_args()
        )
        entity = os.getenv("WANDB_ENTITY")
        if entity:
            init_kwargs["entity"] = entity
        self.wandb.init(**init_kwargs)

    def get_time_str(self):
        return datetime.datetime.now().strftime("%m%d%H%M")

    def get_run_base_name(self):
        name = self.name
        suffix = name.rsplit("_", 1)[-1]
        if len(suffix) == 8 and suffix.isdigit():
            return name[: -(len(suffix) + 1)]
        return name

    def report_scalar(self, name, value, iteration, group_name=None):
        self.report_scalars({name: value}, iteration, group_name)

    def report_scalars(self, values, iteration, group_name=None):
        payload = {}
        for name, value in values.items():
            key = name
            if group_name in {"metrics_val", "metrics_test"}:
                prefix = f"{group_name}/"
                if not key.startswith(prefix):
                    key = prefix + key
            payload[key] = value
        if payload:
            self.wandb.log(payload, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        files = glob.glob(f'{local_path}/*.mp4')
        self.wandb.log({series: [self.wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        self.wandb.config.update(args)  #, allow_val_change=True)  # use allow_val_change ONLY if you want to change existing args (e.g., overwrite)

    def watch_model(self, *args, **kwargs):
        self.wandb.watch(args, kwargs)

    def close(self):
        self.wandb.finish()
