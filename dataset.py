import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from glob import glob


class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        cfg,
        aug,
    ):
        super().__init__()
        self.cfg = cfg
        self.aug = aug
        self.df = df
        self.epoch_len = self.df.shape[0]
        self.sep = "/"
        if hasattr(self.cfg, "kaggle") and self.cfg.kaggle == True:
            self.path_skip = "content/drive/MyDrive/RSNA/Data/PNG/png_full_size_train"
            self.chunk1 = set(
                map(
                    lambda x: x.split("/")[-1],
                    glob("/kaggle/input/rsna-png-chunk1/" + self.path_skip + "/*"),
                )
            )
            # self.chunk2 = set(glob("/kaggle/input/rsna-png-chunk2/" + self.path_skip + "/*"))

        if hasattr(self.cfg, "sep_data_path"):
            self.sep = self.cfg.sep_data_path

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        if hasattr(self.cfg, "kaggle") and self.cfg.kaggle == True:
            if sample.patient_id in self.chunk1:
                img_path = os.path.join(
                    self.cfg.root_dir + "1",
                    self.path_skip,
                    f"{sample.patient_id}{self.sep}{sample.image_id}.png",
                )
            else:
                img_path = os.path.join(
                    self.cfg.root_dir + "2",
                    self.path_skip,
                    f"{sample.patient_id}{self.sep}{sample.image_id}.png",
                )
        else:
            img_path = os.path.join(
                self.cfg.root_dir, f"{sample.patient_id}{self.sep}{sample.image_id}.png"
            )

        label = np.expand_dims(np.array(sample.cancer, dtype=np.int8), axis=0)

        data = {
            "image": img_path,
            "prediction_id": sample.prediction_id,
            "label": label,
        }

        return self.aug(data)

    def __len__(self):
        return self.epoch_len
