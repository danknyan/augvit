# ViT training functions
# imports
import torch
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torch.nn import functional as F
#helpers
from PDD_data_mgmt import CUDAStreamPrefetcher
#visuals
from tqdm import tqdm
# operations
import numpy as np
from time import time
import gc
import joblib
#models
from PDD_ViT import VisionTransformer, FocalLoss, train_flip_transforms, train_mixup_tranforms, train_flipmix_tranforms, og_transforms, val_transforms
from torch import optim
# transforms

# constants
num_classes = 23

# early stopping
# manual implementation of keras.callbacks.EarlyStopping
class EarlyStoppingLoss:
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, start_epoch = 10):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.start_epoch = start_epoch
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        self.current_epoch +=1
        # wait to start counting
        if self.current_epoch < self.start_epoch:
            return False

        if val_loss < self.best_loss - self.min_delta:
            # reset counter and save weights
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        #  stop
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True  
        # or train
        return False  

# training loop
def fit_model(model, 
              optimizer, 
              train_transform, 
              mix_flag, 
              loss_func, 
              epochs, 
              train_loader, 
              val_loader, 
              scheduler, 
              device="cuda"):
    
    scaler = torch.amp.GradScaler("cuda")
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average = "weighted").to(device)
    f1_metric_val = MulticlassF1Score(num_classes=num_classes, average = "weighted").to(device)
    acc_train = MulticlassAccuracy(num_classes=num_classes,average="weighted").to(device)
    acc_val = MulticlassAccuracy(num_classes=num_classes,average="weighted").to(device)
    train_prefetcher = CUDAStreamPrefetcher(train_loader, device)
    val_prefetcher   = CUDAStreamPrefetcher(val_loader,   device)
    training_loss, val_loss = [], []
    train_f1_scores, val_f1_scores = [], []
    train_acc_scores, val_acc_scores = [],[]
    early_stopping = EarlyStoppingLoss()

    for epoch in tqdm(range(epochs), desc="Training Epoch"):
        # train loop
        train_loader.shuffle()
        model.train()
        f1_metric_train.reset()
        acc_train.reset()
        running_loss_tr = torch.tensor(0.0, device=device)

        for data, labels in train_prefetcher:
            # IMG trans
            if mix_flag:
                one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
                data,soft_labels = train_transform(data,one_hot_labels)
            else:
                data = train_transform(data)
            
            optimizer.zero_grad(set_to_none=True)     
            

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_labels = model(data)
                if mix_flag:
                    loss = loss_func(pred_labels.float(),soft_labels)
                    labels = torch.argmax(soft_labels, dim=1)
                else:
                    loss = loss_func(pred_labels.float(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss_tr += loss.detach()
            #batch_loss.append(loss.item())

            preds = torch.argmax(pred_labels, dim=1)
            acc_train.update(preds,labels)
            f1_metric_train.update(preds,labels)


        # train metrics
        training_loss.append((running_loss_tr / len(train_loader)).item())
        train_acc_scores.append(acc_train.compute().detach().cpu())

        train_f1_scores.append(f1_metric_train.compute().detach().cpu())

        # validation loop
        model.eval()
        f1_metric_val.reset()
        acc_val.reset()
        running_loss_val = torch.tensor(0.0, device=device)  
        #batch_val_loss = []
        
        with torch.no_grad():
            for data, labels in val_prefetcher:
                #data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
             
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_labels = model(data)
                    loss = loss_func(pred_labels.float(), labels)

                running_loss_val += loss.detach()

                preds = torch.argmax(pred_labels, dim=1)
                acc_val.update(preds,labels)
                f1_metric_val.update(preds,labels)


        # epoch val metrics
        val_loss.append((running_loss_val / len(val_loader)).item())
        val_acc_scores.append(acc_val.compute().detach().cpu())
        val_f1_scores.append(f1_metric_val.compute().detach().cpu())
        
        # scheduler
        scheduler.step()

        # early stop
        if early_stopping(val_loss[-1], model):
            tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
            break
    del early_stopping, data,labels,
    torch.cuda.empty_cache()
    return (np.array(training_loss), np.array(val_loss),
            np.array(train_f1_scores), np.array(val_f1_scores),
            np.array(train_acc_scores), np.array(val_acc_scores)) 


def fit_model_og(model, 
                 optimizer, 
                 train_transform, 
                 mix_flag, 
                 loss_func, 
                 epochs, 
                 train_loader, 
                 val_loader, 
                 scheduler, 
                 device="cuda"):
    scaler = torch.amp.GradScaler("cuda")
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average = "weighted").to(device)
    f1_metric_val = MulticlassF1Score(num_classes=num_classes, average = "weighted").to(device)
    acc_train = MulticlassAccuracy(num_classes=num_classes,average="weighted").to(device)
    acc_val = MulticlassAccuracy(num_classes=num_classes,average="weighted").to(device)

    training_loss, val_loss = [], []
    train_f1_scores, val_f1_scores = [], []
    train_acc_scores, val_acc_scores = [],[]
    early_stopping = EarlyStoppingLoss()

    for epoch in tqdm(range(epochs), desc="Training Epoch"):
        # train loop
        model.train()
        f1_metric_train.reset()
        acc_train.reset()
        running_loss_tr = torch.tensor(0.0, device=device)

        for data, labels in train_loader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # IMG trans

            data = train_transform(data)
            
            optimizer.zero_grad(set_to_none=True)              

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_labels = model(data)
                loss = loss_func(pred_labels.float(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss_tr += loss.detach()

            preds = torch.argmax(pred_labels, dim=1)
            acc_train.update(preds,labels)
            f1_metric_train.update(preds,labels)


        # train metrics
        training_loss.append((running_loss_tr / len(train_loader)).item())
        train_acc_scores.append(acc_train.compute().detach().cpu())
        train_f1_scores.append(f1_metric_train.compute().detach().cpu())

        # validation loop
        model.eval()
        f1_metric_val.reset()
        acc_val.reset()
        running_loss_val = torch.tensor(0.0, device=device)  
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                data = train_transform(data)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_labels = model(data)
                    loss = loss_func(pred_labels.float(), labels)

                running_loss_val += loss.detach()

                preds = torch.argmax(pred_labels, dim=1)
                acc_val.update(preds,labels)
                f1_metric_val.update(preds,labels)


        # epoch val metrics
        val_loss.append((running_loss_val / len(val_loader)).item())
        val_acc_scores.append(acc_val.compute().detach().cpu())
        val_f1_scores.append(f1_metric_val.compute().detach().cpu())
        
        # scheduler
        scheduler.step()

        # early stop
        if early_stopping(val_loss[-1], model):
            tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
            break
    del early_stopping, data,labels,
    torch.cuda.empty_cache()
    return (np.array(training_loss), np.array(val_loss),
            np.array(train_f1_scores), np.array(val_f1_scores),
            np.array(train_acc_scores), np.array(val_acc_scores)) 

# super long def header, sorry
# train the model and log the results
def train_and_store(
    batch_alias: str, 
    # patch embedding params                 
    patch_size:int,
    # transformer block params
    embedding_dim: int, 
    ffn_inner_dim: int, 
    depth: int,
    attn_head_dim: int,  
    heads: int, 
    head_hidden_dim: int,
    # regularization
    train_aug:str           = "none",
    decay: float            = 0,
    attn_dropout: float     = 0.0,        
    ffn_dropout: float      = 0.1,       
    emb_dropout: float      = 0.1,       
    head_dropout: float     = 0.2,        
    # optim
    opt: str                = "adam",
    lr: float               = 1e-3,
    momentum: float         = None,
    scheduler: bool         = None,
    loss: str               = None,
    # training params
    epochs: int             = 30,
    n_runs: int             = 3,
                   
    verbose: bool           = False,
    save_model: bool        = False,
    return_model: bool      = False,
    train_loader = None,
    val_loader   = None,
    training_log: list[dict] = [],
    device = "cuda"
):

    # label for hyperparameter config
    config_label = (
        f"A={'x'.join(str(l) for l in [heads,attn_head_dim])} "
        f"T={'x'.join(str(l) for l in [depth,ffn_inner_dim])} "
        f"H={head_hidden_dim} "
        f"patch={patch_size} "
        f"aug={train_aug} "
        f"do_e,a,t,h={",".join(str(l) for l in [emb_dropout,attn_dropout,ffn_dropout,head_dropout])} "
        f"opt={opt} lr={lr}"
        + (f" mom={momentum}" if momentum else "")
        + (f" wd={decay}"     if decay    else "")
        + (f" sched=cosine"    if scheduler else "")
    )

    # train loop
    models = []
    for run_id in range(1, n_runs + 1):
        start = time()
        tqdm.write(f"\n[{batch_alias}] run {run_id}/{n_runs} — {config_label}")

        # model config
        model = VisionTransformer(
            patch_size,  
            embedding_dim, 
            ffn_inner_dim, 
            depth,
            attn_head_dim,  
            heads, 
            head_hidden_dim,
            head_dropout,
            attn_dropout, 
            ffn_dropout,
            emb_dropout,
            ).to(device)
     
        # optim
        if opt == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        elif opt == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        else:
            if momentum is None:
                momentum = 0.9
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=decay)
        # loss
        if loss == "focal":
            loss_func = FocalLoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()
            
        # training online augmentation
        if train_aug == "flip":
            train_transform = train_flip_transforms.to(device)
            mix_flag = False
        elif train_aug == "mixup":
            train_transform = train_mixup_tranforms.to(device)
            mix_flag = True
        elif train_aug == "flipmix":
            train_transform = train_flipmix_tranforms.to(device)
            mix_flag = True
        elif train_aug == "og":
            train_transform = og_transforms.to(device)
            mix_flag = False
        else:
            train_transform = val_transforms.to(device)
            mix_flag = False

        active_scheduler = None
        if scheduler:
            active_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=min(10, epochs)
            )

        # fit model
        if train_aug == "og":
            train_model = fit_model_og
        else:
            train_model = fit_model
        
        tr_loss, te_loss, tr_f1, te_f1, tr_acc,te_acc= train_model(
            model, optimizer, train_transform=train_transform, mix_flag=mix_flag,
            scheduler=active_scheduler, loss_func=loss_func,
            epochs=epochs,
            train_loader=train_loader,val_loader=val_loader
        )

        # semi-verbose reporting
        elapsed = (time() - start)
        if verbose:
            tqdm.write(
                f"  run {run_id} | "
                f"epoch {len(tr_loss)} | "
                f"train loss {tr_loss[-1]:.4f} | val loss {te_loss[-1]:.4f} | "
                f"train F1 {tr_f1[-1]:.4f} | val F1 {te_f1[-1]:.4f} | "
                f"train Acc {tr_acc[-1]:.4f} | val Acc {te_acc[-1]:.4f} | "
                f"in {elapsed // 60:.0f}m{elapsed % 60:.0f}s"
            )

        # logging
        training_log.append({
            # identity
            "batch_alias":   batch_alias,
            "config_label":  config_label,
            "run_id":        run_id,
            # hyperparameters
            "patch_size":   patch_size,
            "embedding_dim": embedding_dim, 
            "ffn_inner_dim": ffn_inner_dim, 
            "depth": depth,
            "attn_head_dim": attn_head_dim,  
            "attn_heads": heads, 
            "clf_head_hidden_dim": head_hidden_dim,
            # regularization
            "train_aug":          train_aug,
            "attn_dropout"    : attn_dropout,   
            "ffn_dropout"     : ffn_dropout,        
            "emb_dropout"     : emb_dropout,       
            "head_dropout"    : head_dropout,       
            "weight_decay":  decay,
            # optimization
            "optimizer":     opt,
            "lr":            lr,
            "momentum":      momentum,
            "scheduler":     bool(scheduler),
            "epochs_max":    epochs,
            # results
            "epochs_stopped": len(tr_loss),
            "tr_loss":       list(tr_loss),
            "te_loss":       list(te_loss),
            "tr_acc":        list(tr_acc),
            "te_acc":        list(te_acc),
            "tr_f1":         list(tr_f1),
            "te_f1":         list(te_f1),
            "te_f1_final":   te_f1[-1],
            "te_acc_final":  te_acc[-1],
            "te_loss_final": te_loss[-1],
            "elapsed_min":   elapsed,
        })
        models.append(model)

    # summary across runs
    f1_finals = [r["te_f1_final"] for r in training_log if r["batch_alias"] == batch_alias
              and r["config_label"] == config_label]
    acc_finals = [r["te_acc_final"] for r in training_log if r["batch_alias"] == batch_alias
              and r["config_label"] == config_label]
    tqdm.write(
        f"\n  [{batch_alias}] {config_label}\n"
        f"  val F1 across {n_runs} runs: "
        f"{[f'{v:.4f}' for v in f1_finals]}  "
        f"F1 mean={sum(f1_finals)/len(f1_finals):.4f} "
        f"Acc mean={sum(acc_finals)/len(acc_finals):.4f} "
        f"{elapsed // 60:.0f}m{elapsed % 60:.0f}s"
    )

    #save model
    best_idx = max(range(len(models)),
                    key=lambda i: models[i - 1] and f1_finals[i]) if (save_model or return_model) else None
    joblib.dump(training_log, f"results_{batch_alias}.pkl")
    if save_model:
        save_path = f'./models/{config_label}_best.pth'
        torch.save(models[best_idx].state_dict(), save_path)
        tqdm.write(f"model saved to {save_path}")

    del optimizer, active_scheduler
    gc.collect()

    # return the run with the best final val F1
    if return_model:
        return models[best_idx]
    del models
    torch.cuda.empty_cache()
    gc.collect()
