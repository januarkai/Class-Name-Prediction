import csv
from transformers import TrainerCallback

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs = dict(logs)
        logs['step'] = state.global_step
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(logs.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(logs)
