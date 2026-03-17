import os
import torch
import pandas as pd
from config.exp_config import Config
from helper.logger import get_logger
from core.TVDiag import TVDiag
from process.EventProcess import EventProcess
from core.model.MainModel import MainModel

cfg = Config("gaia")
logger = get_logger("./logs/gaia", "inspect_preds")
processor = EventProcess(cfg, logger)
train_data, aug_data, test_data = processor.process(reconstruct=cfg.reconstruct)

diag = TVDiag(cfg, logger, "./logs/gaia")
ckpt = torch.load("./logs/gaia/TVDiag.pt", map_location=diag.device)
model = MainModel(cfg).to(diag.device)
model.load_state_dict(ckpt["model"])
model.eval()

rows = []
with torch.no_grad():
    for i, (graph, labels) in enumerate(test_data):
        graph = graph.to(diag.device)
        _, _, root_logit, type_logit = model(graph)

        type_true = int(labels[1])
        type_pred = int(type_logit.argmax(dim=1).item())

        root_true = int(graph.ndata["root"].argmax().item())
        root_pred = int(root_logit.squeeze(-1).argmax().item())

        rows.append({
            "sample_id": i,
            "type_true": type_true,
            "type_pred": type_pred,
            "root_true_local_idx": root_true,
            "root_pred_local_idx": root_pred,
        })

df = pd.DataFrame(rows)
out = "./logs/gaia/predictions_vs_target.csv"
df.to_csv(out, index=False)
print("saved:", out)