import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_array, seq_len=32, pred_len=32,
                 level_idx=0, future_rain_idx=4, vwc_idx=9):
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.level_idx = level_idx
        self.future_rain_idx = future_rain_idx
        self.vwc_idx = vwc_idx

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_hist = self.data[idx : idx + self.seq_len]  # [L, F]

        fut_block = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len
        ]  # [P, F]

        fut_seq = fut_block[:, self.future_rain_idx].unsqueeze(-1)  # [P,1]
        y = fut_block[:, self.level_idx]                            # [P]
        vwc_future = fut_block[:, self.vwc_idx]                     # [P]

        return x_hist, fut_seq, y, vwc_future
