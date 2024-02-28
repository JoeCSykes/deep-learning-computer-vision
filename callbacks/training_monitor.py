from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import os
import json


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path=None, json_path=None, start_at=0, start_graph_at=0):
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self. start_at = start_at
        self.start_graph_at = start_graph_at
        self.H = {}

    def on_train_begin(self, logs={}):

        if self.json_path:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # trim any entries that are past the starting epoch
                if self.start_at > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            log = self.H.get(k, [])
            log.append(v)
            self.H[k] = log

        if self.json_path:
            with open(self.json_path, "w") as file:
                file.write(json.dumps(self.H))

        if len(self.H["loss"]) > self.start_graph_at + 1:
            x_val = np.arange(0, len(self.H["loss"][self.start_graph_at:]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(x_val, self.H["loss"][self.start_graph_at:], label="train_loss")
            plt.plot(x_val, self.H["val_loss"][self.start_graph_at:], label="val_loss")
            plt.plot(x_val, self.H["accuracy"][self.start_graph_at:], label="train_acc")
            plt.plot(x_val, self.H["val_accuracy"][self.start_graph_at:], label="val_acc")
            plt.title(f"Training Loss and Accuracy [Epoch {len(self.H['loss'])}]")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss / Accuracy")
            plt.legend()

            if self.fig_path:
                plt.savefig(self.fig_path)
                plt.close()
            else:
                plt.show()
