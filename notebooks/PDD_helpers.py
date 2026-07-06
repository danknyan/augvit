# Helper Functions for EDA, Visualization, and Dataloading 
#imports 
# ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# data mgmt
from collections import defaultdict
from dataclasses import dataclass
from torch.utils.data import Subset
# torch backend
from torchvision.transforms import v2
import torch
from PDD_ViT import val_transforms
from torch.utils.data import DataLoader
# metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
# visual
from IPython.display import display

# num classes
num_classes = 23

# Vit config storage
@dataclass
class ViTConfig:
    batch_alias: str
    # architecture
    patch_size: int        = 28
    embedding_dim: int     = 512
    depth: int             = 6
    heads: int             = 8
    attn_head_dim: int     = 64
    ffn_inner_dim: int     = 1024
    head_hidden_dim: int   = 256
    # regularization
    emb_dropout: float     = 0.0
    attn_dropout: float    = 0.0
    ffn_dropout: float     = 0.0
    head_dropout: float    = 0.0
    decay: float           = 1e-5
    # optimization
    opt: str               = "adamw"
    lr: float              = 1e-4
    scheduler: bool        = True
    train_aug: str         = "none"
    loss: str              = "focal"
    # training
    epochs: int            = 75
    n_runs: int            = 3

# master transforms list
NORMALIZE = v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
DENORMALIZE = v2.Normalize(
    mean=[-m/s for m, s in zip(NORMALIZE.mean, NORMALIZE.std)],
    std=[1/s for s in NORMALIZE.std]
)

# EDA Helpers

# get class counts per dataset
def class_counts(y, class_names):
    counts = np.bincount(y, minlength=len(class_names))
    return counts

# Inspect show the dataset class names, dataset dimensions
def inspect_sample(dataset):
    image, label = dataset[0]
    num_classes = len(dataset.class_names)
    class_names = dataset.class_names
    labels = dataset.labels.tolist()
    print(f"Image dimensions: {image.shape}")
    print(f"Total classes: {num_classes}")
    print(f"Total images: {len(labels)}")
    # Class names
    print(f"* {", ".join([n if ind-1 < 0 or class_names[ind-1][:2] == n[:2] else "\n* "+n for ind,n in enumerate(class_names)])}")

# show sample image
def show_tensor_image(sample_ind:int, dataset, pred_label=None):
    img_tensor, label = dataset[sample_ind]

    image = (img_tensor.clone().detach().cpu().float() / 255.0)
    image = image.clamp(0, 1).numpy().transpose(1, 2, 0)

    if pred_label is not None:                     
        title = (f"True Label: {dataset.class_names[label]} ({label})\n"
                    f"Prediction: {dataset.class_names[pred_label]} ({int(pred_label)})")
    else:
        title = f"True Label: {dataset.class_names[label]} ({label})"
    plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# visualize class balance
def plot_class_balance(augmented_dataset:DataLoader, 
                       original_dataset:DataLoader,
                       original_classlist_path:str = "./FieldPlant-11/_classes.csv"):
    df = pd.read_csv(original_classlist_path)
    df.columns = df.columns.str.strip()

    class_names = augmented_dataset.class_names
    labels = augmented_dataset.labels.tolist()
    labels_og = df[class_names].values.argmax(axis=1)
    aug_counts = class_counts(labels, augmented_dataset.class_names)
    og_counts = class_counts(labels_og, original_dataset.class_names)

    x = np.arange(len(class_names))
    width = 0.4

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x, aug_counts, width, label="Augmented", color="#4C72B0")
    ax.bar(x + width/2.3, og_counts*.997,  width, label="Original",  color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Sample count")
    ax.set_title("Class Balance")
    ax.legend()
    plt.tight_layout()
    plt.show()

# plot confusion matrix
def plot_cm(model, dataloader, val_transforms=val_transforms,device = "cuda"):    
    # get predictions
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            data = val_transforms(data)
            all_preds.extend(model(data).argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # score, plot
    acc = balanced_accuracy_score(all_labels,all_preds)
    f1 = f1_score(all_labels,all_preds, average="weighted")
    print(f"ACC: {acc:.4f} \n F1-W] {f1:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# plot loss, f1, accuracy of epochs
def plot_training(batch_alias: str, log: list[dict] = None, show_runs: bool = False):

    rows = [r for r in log if r["batch_alias"] == batch_alias]
    if not rows:
        print(f"No runs found for batch_alias='{batch_alias}'")
        return

    # group by config_label
    groups = defaultdict(list)
    for r in rows:
        groups[r["config_label"]].append(r)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6+len(rows)*.5))
    fig.suptitle(f"{batch_alias} — {'all runs' if show_runs else 'mean ± std'}",
                 fontsize=14, fontweight="bold")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, run_list) in enumerate(groups.items()):
        color = colors[i % len(colors)]

        if show_runs:
            for r in run_list:
                eps = np.arange(1, len(r["tr_loss"]) + 1)
                ax1.plot(eps, r["tr_loss"], linestyle="--", color=color, alpha=0.5)
                ax1.plot(eps, r["te_loss"],                color=color, alpha=0.5)
                ax2.plot(eps, r["tr_f1"],   linestyle="--", color=color, alpha=0.5)
                ax2.plot(eps, r["te_f1"],                  color=color, alpha=0.5)
                ax3.plot(eps, r["tr_acc"],   linestyle="--", color=color, alpha=0.5)
                ax3.plot(eps, r["te_acc"],                  color=color, alpha=0.5,
                         label=f"{label} run {r['run_id']}")
        else:
            # pad sequences to the same length, then average
            max_len = max(len(r["tr_loss"]) for r in run_list)
            def pad(seq): return seq + [seq[-1]] * (max_len - len(seq))

            tr_loss_arr = np.array([pad(r["tr_loss"]) for r in run_list])
            te_loss_arr = np.array([pad(r["te_loss"]) for r in run_list])
            tr_f1_arr   = np.array([pad(r["tr_f1"])   for r in run_list])
            te_f1_arr   = np.array([pad(r["te_f1"])   for r in run_list])
            tr_acc_arr   = np.array([pad(r["tr_acc"])   for r in run_list])
            te_acc_arr   = np.array([pad(r["te_acc"])   for r in run_list])
            eps = np.arange(1, max_len + 1)

            for arr, ax, ls in [
                (tr_loss_arr, ax1, "--"), (te_loss_arr, ax1, "-"),
                (tr_f1_arr,   ax2, "--"), (te_f1_arr,   ax2, "-"),
                (tr_acc_arr,   ax3, "--"), (te_acc_arr,   ax3, "-"),
            ]:
                mean, std = arr.mean(axis=0), arr.std(axis=0)
                ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15)

            ax1.plot(eps, tr_loss_arr.mean(0), "--", color=color, label=f"{label} — Train")
            ax1.plot(eps, te_loss_arr.mean(0),       color=color, label=f"{label} — Val")
            ax2.plot(eps, tr_f1_arr.mean(0),   "--", color=color, label=f"{label} — Train")
            ax2.plot(eps, te_f1_arr.mean(0),         color=color, label=f"{label} — Val")
            ax3.plot(eps, tr_acc_arr.mean(0),   "--", color=color, label=f"{label} — Train")
            ax3.plot(eps, te_acc_arr.mean(0),         color=color, label=f"{label} — Val")

    for ax, title, ylabel in [
        (ax1, "Loss",            "Loss"),
        (ax2, "F1 Score (Weighted)", "F1"),
        (ax3, "Accuracy Score (Weighted)", "Acc"),
    ]:
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(0.5, -0.3),loc="upper center", borderaxespad=0)
    plt.tight_layout()
    fig.savefig(f"{batch_alias}.png")
    plt.show()

# show training results as leaderboard
def show_lb(log: list[dict] = None, 
            top_n: int = 10, 
            batch_alias: str = None, 
            return_df = False):

    rows = [r for r in log if batch_alias is None or r["batch_alias"] == batch_alias]
    if not rows:
        print("No runs found.")
        return

    df = pd.DataFrame(rows)

    # aggregate
    agg = (
        df.groupby(["config_label"])
        .agg(
            runs        = ("run_id",       "count"),
            f1_mean     = ("te_f1_final",  "mean"),
            f1_std      = ("te_f1_final",  "std"),
            acc_mean     = ("te_acc_final",  "mean"),
            acc_std      = ("te_acc_final",  "std"),
            loss_mean   = ("te_loss_final","mean"),
            epochs_mean = ("epochs_stopped","mean"),
            train_time_mean = ("elapsed_min", "mean"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    agg.index += 1
    agg["f1_std"]  = agg["f1_std"].fillna(0)
    agg["f1 (mean)"]  = agg["f1_mean"].map("{:.4f}".format)
    agg["acc_std"]  = agg["acc_std"].fillna(0)
    agg["acc (mean)"]  = agg["acc_mean"].map("{:.4f}".format)
    #agg["f1 (mean ± std)"] = agg.apply(
    #    lambda r: f"{r.f1_mean:.4f} ± {r.f1_std:.4f}", axis=1
    #)
    agg["loss (mean)"]  = agg["loss_mean"].map("{:.4f}".format)
    agg["epochs (mean)"]= agg["epochs_mean"].map("{:.1f}".format)
    agg["training time (mean)"]= agg["train_time_mean"].map("{:.1f}".format)

    display(
        agg[["config_label","runs","f1 (mean)","acc (mean)","loss (mean)","epochs (mean)","training time (mean)"]]
        .rename(columns={"batch_alias": "sweep", "config_label": "config"})
        .style
        .set_caption("Leaderboard — ranked by val F1")
        .background_gradient(subset=["f1 (mean)"], cmap="YlGn")
        .set_properties(**{"text-align": "left"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "left"), ("font-weight", "500")]},
            {"selector": "caption", "props": [("font-weight", "500"), ("font-size", "14px"),
                                              ("text-align", "left"), ("padding-bottom", "8px")]},
        ])
    )
    if return_df:
        return agg




# show incorrect predictions
def show_predictions(
    model,
    dataset,
    n: int = 4,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    cols: int = 4,
    figsize_per_cell=(1.4, 1.7),
    device = "cuda",
):

    model = model.to(device).eval()
 
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t  = torch.tensor(std ).view(1, 3, 1, 1)
    class_names = dataset.class_names
 
    mistake_images = []
    mistake_preds  = []
    mistake_labels = []
 
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
 
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds  = logits.argmax(dim=1).cpu()
 
            wrong_mask = preds != labels
            mistake_images.append(images[wrong_mask])
            mistake_preds .append(preds  [wrong_mask])
            mistake_labels.append(labels [wrong_mask])
 
            if sum(len(m) for m in mistake_images) >= n:
                break
 
    if not mistake_images:
        print("No mistakes found.")
        return
 
    all_images = torch.cat(mistake_images)[:n]
    all_preds  = torch.cat(mistake_preds )[:n]
    all_labels = torch.cat(mistake_labels)[:n]
 
    imgs_display = (all_images * std_t + mean_t).clamp(0, 1)
 
    total = len(all_images)
    rows  = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_cell[0] * cols,
                                      figsize_per_cell[1] * rows))
    axes = np.array(axes).reshape(-1)
 
    for i, ax in enumerate(axes):
        if i < total:
            img = imgs_display[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"Pred: {class_names[all_preds[i].item()]}\n"
                f"True: {class_names[all_labels[i].item()]}",
                fontsize=6,
                color="#e74c3c",
                pad=1,
            )
        else:
            ax.axis("off")
 
    fig.suptitle(
        f"Example Mistaken Predictions",
        fontsize=11,
        y=0.95,
    )
    plt.tight_layout()
    plt.show()

# dataset partitioning and dataset management
## split data
def get_train_val_test_datasets(dataset, train_p =0.8,val_test_mix = 0.5):

    indices = np.arange(len(dataset))
    idx_labels = dataset.labels.tolist()

    idx_train, idx_temp, _, y_temp = train_test_split(
        indices, idx_labels,
        test_size=1-train_p,
        stratify=idx_labels,
        random_state=42
    )

    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp,
        test_size=val_test_mix,
        stratify=y_temp,
        random_state=42
    )
    train_dataset = Subset(dataset, idx_train)
    val_dataset   = Subset(dataset, idx_val)
    test_dataset  = Subset(dataset, idx_test)
    return train_dataset,val_dataset,test_dataset
