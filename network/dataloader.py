import dgl
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from utils.Config import Config
from utils.soft_label import build_ordinal_soft_label


class KGNetDataset(Dataset):
    def __init__(self, data_type: str, cfg: Config, topk=None):
        super(KGNetDataset, self).__init__()
        self.cfg        = cfg
        self.topk       = topk
        self.label_mode = cfg.label_mode
        self.f = np.loadtxt(
            f"{cfg.index_folder}/{data_type}_{cfg.fold}.csv", dtype=str
        )
        # For multi_label mode on MRNet: load the secondary label CSV.
        # Expects files:  <index_folder>/multi_<task>_<fold>.csv
        # Each line: <path>,<abnormality_label>,<acl_label>
        # Falls back gracefully if the file does not exist (non-MRNet datasets).
        self.multi_label_data = None
        if self.label_mode == "multi_label":
            ml_path = f"{cfg.index_folder}/multi_{data_type}_{cfg.fold}.csv"
            try:
                self.multi_label_data = np.loadtxt(ml_path, dtype=str)
            except OSError:
                # File not found: multi_label will be [0, 0] as placeholder.
                pass

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        data = np.load(f"{self.cfg.path}/{self.f[idx].split(',')[0]}.npz", allow_pickle=True)
        grade = int(self.f[idx].split(",")[1])

        patch = data["org"].item()

        patch_sag, m_sag, s_sag = self.norm_img(patch[self.cfg.views[0]])
        patch_cor, m_cor, s_cor = self.norm_img(patch[self.cfg.views[1]])
        patch_axi, m_axi, s_axi = self.norm_img(patch[self.cfg.views[2]])
        patch = torch.stack([patch_sag, patch_cor, patch_axi], dim=1)
        mean = [m_sag, m_cor, m_axi]
        std = [s_sag, s_cor, s_axi]

        seg = data["seg"].item()
        seg = torch.stack([seg[self.cfg.views[0]], seg[self.cfg.views[1]], seg[self.cfg.views[2]]], dim=1).type(torch.long)
        # seg = rearrange(seg, "n v h w -> (n v) h w")
        les = torch.tensor([0], dtype=torch.long) # not used
        grade = torch.tensor(grade, dtype=torch.long)
        pos = torch.tensor(data["pos"], dtype=torch.float32)

        if self.topk is None:
            graph = data["graph"].item()
        else:
            graph = self.create_knn_graph(pos, self.topk)

        pos = self.norm_pos(pos)
        name = self.f[idx]

        # --- Soft label (ordinal prior) ---
        soft_label = None
        if self.label_mode == "soft":
            soft_label = build_ordinal_soft_label(
                int(grade.item()),
                num_classes=self.cfg.num_cls,
                alpha=self.cfg.ordinal_alpha,
            )
            soft_label = torch.tensor(soft_label, dtype=torch.float32)

        # --- Multi-label vector (MRNet joint tasks) ---
        multi_label = None
        if self.label_mode == "multi_label":
            if self.multi_label_data is not None:
                row = self.multi_label_data[idx]
                multi_label = torch.tensor(
                    [float(row[1]), float(row[2])], dtype=torch.float32
                )
            else:
                # Placeholder if CSV not found
                multi_label = torch.zeros(self.cfg.num_labels, dtype=torch.float32)

        return (graph, patch, seg, pos, grade, les, name, mean, std,
                soft_label, multi_label)

    def norm_pos(self, pos):
        for i in range(3):
            pmax = pos[:, i].max()
            pmin = pos[:, i].min()
            pos[:, i] = (pos[:, i] - pmin) / (pmax - pmin)
        return pos

    def norm_img(self, patch):
        # print(
        #     f"check patch {patch.max()}, {patch.min()}, {patch.mean()}, {patch.std()}"
        # )
        mean = torch.mean(patch)
        std = torch.std(patch)
        patch = (patch - mean) / std
        # print(
        #     f"check patch {patch.max()}, {patch.min()}, {patch.mean()}, {patch.std()}\n"
        # )
        return patch, mean, std

    def create_knn_graph(self, points, k):
        # points: [n, 3] tensor containing the coordinates of the points
        n = points.shape[0]
        
        # Compute pairwise Euclidean distance
        dist_sq = torch.cdist(points, points, p=2)
        
        # Find the indices of k nearest neighbors (excluding itself)
        # Using topk to find the smallest k+1 distances, then exclude the first column (self-loops)
        _, indices = torch.topk(dist_sq, k=k, largest=False, sorted=False)
        # indices = indices[:, 1:]  # Remove self index
        
        # Create source and destination nodes for each edge
        src = torch.arange(n).repeat_interleave(k)
        dst = indices.flatten()
        
        # Create the DGL graph
        graph = dgl.graph((src, dst))
        
        return graph


class KGNetData(object):
    def __init__(self):
        super().__init__()
        self.patch      = None
        self.pos        = None
        self.les        = None
        self.seg        = None
        self.graph      = None
        self.grade      = None
        self.name       = None
        self.soft_label  = None   # float [B, num_cls]  — populated in soft mode
        self.multi_label = None   # float [B, num_labels] — populated in multi_label mode
        self.label_mode  = "single"

        self.N = -1
        self.keep = None
        self.mask = None
        self.idx_rec = None
        self.idx_mix = None
        self.idx_seg = None
        self.device = "cpu"
        self.mean = None
        self.std = None
        return

    def to(self, device):
        self.patch = self.patch.to(device)
        self.pos   = self.pos.to(device)
        self.seg   = self.seg.to(device)
        self.les   = self.les.to(device)
        self.graph = self.graph.to(device)
        self.grade = self.grade.to(device)
        if self.soft_label is not None:
            self.soft_label = self.soft_label.to(device)
        if self.multi_label is not None:
            self.multi_label = self.multi_label.to(device)
        return


def collate(samples):
    _data = KGNetData()
    samples = list(filter(lambda x: x is not None, samples))
    graphs, patch, seg, pos, grade, les, names, mean, std, soft_labels, multi_labels = \
        map(list, zip(*samples))
    _data.patch  = torch.cat(patch, dim=0)
    _data.seg    = torch.cat(seg, dim=0)
    _data.les    = torch.cat(les, dim=0)
    _data.pos    = torch.cat(pos, dim=0)
    _data.graph  = dgl.batch(graphs)
    _data.grade  = torch.tensor(grade, dtype=torch.long)
    _data.name   = names
    _data.N      = _data.patch.shape[0]
    _data.mean   = mean[0]
    _data.std    = std[0]

    # Soft labels (None when label_mode != 'soft')
    if soft_labels[0] is not None:
        _data.soft_label = torch.stack(soft_labels, dim=0)

    # Multi-labels (None when label_mode != 'multi_label')
    if multi_labels[0] is not None:
        _data.multi_label = torch.stack(multi_labels, dim=0)

    return _data


def KGNetDataloader(cfg: Config, topk=None):
    dataset = {}
    type_list = ["train", 'valid', "test",]
    shuffles = [True, False, False]
    for item, shuffle in zip(type_list, shuffles):
        dataset[item] = DataLoader(
            KGNetDataset(data_type=item, cfg=cfg, topk=topk),
            batch_size=cfg.bs,
            collate_fn=collate,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
        )
    return dataset
