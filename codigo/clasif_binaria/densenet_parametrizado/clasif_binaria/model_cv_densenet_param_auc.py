import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json, nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import DenseNet121
from monai.transforms import Compose, EnsureType

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from collections import defaultdict


#config
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
# WEIGHT_DECAY = 1e-5

# Ruta de resultados incremental
RUN_TAG = time.strftime("%Y%m%d-%H%M%S")
RESULTS_CSV = f"resultados_modelo_densenet_param_1_mask_reforzando_nodulo_early_stop{RUN_TAG}.csv"

def append_rows_to_csv(rows, csv_path=RESULTS_CSV):
    """Escribe filas (lista de dicts) al CSV, añadiendo cabecera solo si no existe."""
    df_rows = pd.DataFrame(rows)
    file_exists = os.path.exists(csv_path)
    df_rows.to_csv(csv_path, mode="a", header=not file_exists, index=False)


#leer datos
df = pd.read_csv("./../clinical_data/clinical_data.csv", na_values="NaN")
labels_dict_numeric = {
    k: 1 if str(v).strip().lower() in ['sí', 'si', 'yes', '1', 'true'] else 0
    for k, v in dict(zip(df['patient_id'], df['Complicación'])).items()
}

# #crop al bbox del pulmon
# def crop_to_bbox(vol, mask, margin=5):
#     if mask.sum() == 0: 
#         return vol  # nada que recortar
#     z,y,x = np.where(mask>0)
#     z0, z1 = max(z.min()-margin, 0), min(z.max()+1+margin, vol.shape[0])
#     y0, y1 = max(y.min()-margin, 0), min(y.max()+1+margin, vol.shape[1])
#     x0, x1 = max(x.min()-margin, 0), min(x.max()+1+margin, vol.shape[2])
#     return vol[z0:z1, y0:y1, x0:x1]

#gradcam
def get_last_conv_layer(model):
    for layer in reversed(model.features):
        if isinstance(layer, nn.Conv3d):
            return layer
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, nn.Conv3d):
                    return sub
    raise ValueError("No se encontró una capa Conv3d válida en el modelo")


# def apply_gradcam(model, volume, label, device, output_base_dir, example_id=""):
#     model.train()
#     target_layers = [get_last_conv_layer(model)]
#     cam = GradCAM(model=model, target_layers=target_layers)

#     input_tensor = volume.to(device)
#     targets = [ClassifierOutputTarget(label)]

#     with torch.enable_grad():
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

#     for idx in range(0, grayscale_cam.shape[0], max(1, grayscale_cam.shape[0] // 6)):
#         plt.figure(figsize=(6, 6))
#         img = input_tensor.cpu().numpy()[0, 0, idx]
#         norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#         heatmap = grayscale_cam[idx]
#         rgb_img = np.repeat(norm_img[..., np.newaxis], 3, axis=-1)
#         result = show_cam_on_image(rgb_img.astype(np.float32), heatmap, use_rgb=True)

#         os.makedirs(output_base_dir, exist_ok=True)
#         path = os.path.join(output_base_dir, f"{example_id}_slice_{idx}.png")
#         plt.imsave(path, result)
#         plt.close()


def apply_gradcam(model, volume, label, device, output_base_dir, example_id=""):
    model.eval()
    target_layers = [get_last_conv_layer(model)]
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = volume.to(device)  # (1, 1, D, H, W)
    targets = [ClassifierOutputTarget(label)]

    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (D,H,W)

    # Enmascara CAM con pulmón (del propio input ya enmascarado)
    vol_np = input_tensor.detach().cpu().numpy()
    lung_mask = (vol_np[0, 0] > 0).astype(np.float32)   # (D,H,W)
    grayscale_cam = grayscale_cam * lung_mask

    os.makedirs(output_base_dir, exist_ok=True)

    # Guardar aprox 6 cortes equiespaciados
    D = grayscale_cam.shape[0]
    step = max(1, D // 6)
    for idx in range(0, D, step):
        base_img = vol_np[0, 0, idx]
        # normaliza a [0,1]
        norm_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
        rgb_img = np.repeat(norm_img[..., None], 3, axis=-1).astype(np.float32)

        heatmap = grayscale_cam[idx]
        result = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

        out_path = os.path.join(output_base_dir, f"{example_id}_slice_{idx}.png")
        plt.imsave(out_path, result)



#visualiza solo unos cuantos
# def visualizar_gradcams(model, dataset, val_idx, device, output_base_dir):
#     print("Generando ejemplos de Grad-CAM...")
#     correctos = defaultdict(list)
#     incorrectos = defaultdict(list)

#     loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
#     model.eval()

#     for i, (vol, label) in enumerate(loader):
#         vol = vol.to(device)
#         real = label.item()
#         pred = torch.argmax(model(vol), dim=1).item()

#         patient_id = dataset.patients[val_idx[i]]

#         if pred == real:
#             correctos[real].append( (vol, patient_id, pred) )
#         else:
#             incorrectos[real].append( (vol, patient_id, pred) )

#     # Generar hasta 2 ejemplos por clase
#     for clase in [0, 1]:
#         for (v, pid, pred) in correctos[clase][:2]:
#             example_id = f"label{clase}_correcto_pred{pred}_pid{pid}"
#             apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)

#         for (v, pid, pred) in incorrectos[clase][:2]:
#             example_id = f"label{clase}_fallo_pred{pred}_pid{pid}"
#             apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)

def visualizar_gradcams(model, dataset, val_idx, device, output_base_dir):
    print("Generando ejemplos de Grad-CAM...")
    correctos = defaultdict(list)
    incorrectos = defaultdict(list)

    loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
    model.eval()

    for i, (vol, label) in enumerate(loader):
        vol = vol.to(device)
        real = label.item()
        pred = torch.argmax(model(vol), dim=1).item()

        patient_id = dataset.patients[val_idx[i]]

        if pred == real:
            correctos[real].append( (vol, patient_id, pred) )
        else:
            incorrectos[real].append( (vol, patient_id, pred) )

    #  generamos todos los ejemplos 
    for clase in [0, 1]:
        for (v, pid, pred) in correctos[clase]:
            example_id = f"label{clase}_correcto_pred{pred}_pid{pid}"
            apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)

        for (v, pid, pred) in incorrectos[clase]:
            example_id = f"label{clase}_fallo_pred{pred}_pid{pid}"
            apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)


#dataset
# class LungCTDataset(Dataset):
#     def __init__(self, data_dir, mask_dir, labels_dict, transform=None):
#         self.data_dir = data_dir
#         self.mask_dir = mask_dir
#         self.patients = list(labels_dict.keys())
#         self.labels = labels_dict
#         self.transform = transform

#     def __len__(self):
#         return len(self.patients)

#     def __getitem__(self, idx):
#         patient_id = self.patients[idx]
#         volume = np.load(os.path.join(self.data_dir, f"{patient_id}.npy"))

#         mask_path = os.path.join(self.mask_dir, f"{patient_id}.npy")
#         if os.path.exists(mask_path):
#             mask = np.load(mask_path).astype(bool)
#             if mask.shape == volume.shape:
#                 volume = volume * mask

#         volume = np.expand_dims(volume, axis=0)
#         if self.transform:
#             volume = self.transform(volume)
#         label = self.labels[patient_id]
#         return volume, torch.tensor(label, dtype=torch.long)

#dataset with the 3 masks
# class LungCTTripletNPYDataset(Dataset):
#     """
#     root_npy_dir/
#       images/{PID}.npy         -> (D,H,W)
#       masks_lung/{PID}.npy     -> (D,H,W)
#       masks_nodule/{PID}.npy   -> (D,H,W)
#     """
#     def __init__(self, root_npy_dir, labels_dict, patients=None, transform=None):
#         self.root = root_npy_dir
#         self.images_dir = os.path.join(root_npy_dir, "images")
#         self.lung_dir   = os.path.join(root_npy_dir, "masks_lung")
#         self.nod_dir    = os.path.join(root_npy_dir, "masks_nodule")
#         self.labels = labels_dict
#         self.patients = patients if patients is not None else list(labels_dict.keys())
#         self.transform = transform

#         # Filtra por ficheros existentes
#         self.patients = [pid for pid in self.patients
#                          if all(os.path.exists(os.path.join(d, f"{pid}.npy"))
#                                 for d in [self.images_dir, self.lung_dir, self.nod_dir])]

#     def __len__(self): return len(self.patients)

#     def __getitem__(self, idx):
#         pid = self.patients[idx]
#         img  = np.load(os.path.join(self.images_dir, f"{pid}.npy")).astype(np.float32)   # (D,H,W)
#         lung = np.load(os.path.join(self.lung_dir,   f"{pid}.npy")).astype(np.float32)   # (D,H,W)
#         nod  = np.load(os.path.join(self.nod_dir,    f"{pid}.npy")).astype(np.float32)   # (D,H,W)

#         # Asegura binarización por si vinieran con valores raros
#         lung = (lung > 0).astype(np.float32)
#         nod  = (nod  > 0).astype(np.float32)

#         vol = np.stack([img, lung, nod], axis=0)  # (3,D,H,W)

#         if self.transform:
#             vol = self.transform(vol)
#         label = torch.tensor(self.labels[pid], dtype=torch.long)
#         return vol, label

# class LungCTTripletNPYDataset(Dataset):
#     """
#     root_npy_dir/
#       images/{PID}.npy         -> (D,H,W)  imagen ya windowed/normalizada [0,1] en el preprocesado
#       masks_lung/{PID}.npy     -> (D,H,W)  binaria {0,1}
#       masks_nodule/{PID}.npy   -> (D,H,W)  binaria {0,1}
#     """
#     def __init__(self, root_npy_dir, labels_dict, patients=None, transform=None):
#         self.root = root_npy_dir
#         self.images_dir = os.path.join(root_npy_dir, "images")
#         self.lung_dir   = os.path.join(root_npy_dir, "masks_lung")
#         self.nod_dir    = os.path.join(root_npy_dir, "masks_nodule")
#         self.labels = labels_dict
#         self.patients = patients if patients is not None else list(labels_dict.keys())
#         self.transform = transform

#         # Filtra por ficheros existentes
#         self.patients = [pid for pid in self.patients
#                          if all(os.path.exists(os.path.join(d, f"{pid}.npy"))
#                                 for d in [self.images_dir, self.lung_dir, self.nod_dir])]

#     def __len__(self): 
#         return len(self.patients)

#     def __getitem__(self, idx):
#         pid = self.patients[idx]
#         # Carga
#         img  = np.load(os.path.join(self.images_dir, f"{pid}.npy")).astype(np.float32)   # (D,H,W)
#         lung = np.load(os.path.join(self.lung_dir,   f"{pid}.npy")).astype(np.float32)   # (D,H,W)
#         nod  = np.load(os.path.join(self.nod_dir,    f"{pid}.npy")).astype(np.float32)   # (D,H,W)

#         # Asegura binarización
#         lung = (lung > 0.5).astype(np.float32)
#         nod  = (nod  > 0.5).astype(np.float32)

#         # Canal 0: imagen enmascarada por pulmón (fuera del pulmón = 0)
#         img_masked = img * lung

#         # Apila canales: (3, D, H, W)
#         vol = np.stack([img_masked, lung, nod], axis=0).astype(np.float32)

#         # MONAI transform
#         if self.transform:
#             vol = self.transform(vol)

#         label = torch.tensor(self.labels[pid], dtype=torch.long)
#         return vol, label


#solo 1 canal reforzando la parte del nodulo
class LungCTMaskedNPYDataset(Dataset):
    """
    root_npy_dir/
      images/{PID}.npy         -> (D,H,W)  imagen windowed/normalizada [0,1] en preprocesado
      masks_lung/{PID}.npy     -> (D,H,W)  binaria {0,1}
      masks_nodule/{PID}.npy   -> (D,H,W)  binaria {0,1}
    """
    def __init__(self, root_npy_dir, labels_dict, patients=None, transform=None,
                 emphasize_nodule=False, nodule_gain=0.0):
        """
        emphasize_nodule: si True, multiplica intensidades intrapulmonares por (1 + nodule_gain*mask_nodule)
        nodule_gain:     0.0 -> sin énfasis;  por ejemplo 0.2 ó 0.5 para realzar nódulo
        """
        self.root = root_npy_dir
        self.images_dir = os.path.join(root_npy_dir, "images")
        self.lung_dir   = os.path.join(root_npy_dir, "masks_lung")
        self.nod_dir    = os.path.join(root_npy_dir, "masks_nodule")
        self.labels = labels_dict
        self.patients = patients if patients is not None else list(labels_dict.keys())
        self.transform = transform
        self.emphasize_nodule = emphasize_nodule
        self.nodule_gain = float(nodule_gain)

        # Filtra por ficheros existentes
        self.patients = [pid for pid in self.patients
                         if all(os.path.exists(os.path.join(d, f"{pid}.npy"))
                                for d in [self.images_dir, self.lung_dir, self.nod_dir])]

    def __len__(self): 
        return len(self.patients)

    def __getitem__(self, idx):
        pid  = self.patients[idx]
        img  = np.load(os.path.join(self.images_dir, f"{pid}.npy")).astype(np.float32)   # (D,H,W) en [0,1]
        lung = np.load(os.path.join(self.lung_dir,   f"{pid}.npy")).astype(np.float32)
        nod  = np.load(os.path.join(self.nod_dir,    f"{pid}.npy")).astype(np.float32)

        # Binariza antes de calcular bbox
        lung_bin = (lung > 0.5).astype(np.uint8)
        nod_bin  = (nod  > 0.5).astype(np.uint8)

        # === bbox común desde la máscara de pulmón ORIGINAL (sin recortes previos) ===
        # margin = 5
        # if lung_bin.sum() > 0:
        #     z, y, x = np.where(lung_bin > 0)
        #     z0, z1 = max(z.min() - margin, 0), min(z.max() + 1 + margin, img.shape[0])
        #     y0, y1 = max(y.min() - margin, 0), min(y.max() + 1 + margin, img.shape[1])
        #     x0, x1 = max(x.min() - margin, 0), min(x.max() + 1 + margin, img.shape[2])
        #     sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        # else:
        #     # sin pulmón → no recortes
        #     sl = (slice(None), slice(None), slice(None))

        # Aplica exactamente el mismo recorte a los tres volúmenes
        # img  = img[sl]
        # lung = lung_bin[sl].astype(np.float32)  # ya binaria en [0,1]
        # nod  = nod_bin[sl].astype(np.float32)

        # Enmascara imagen por pulmón
        img_masked = img * lung

        # Realce del nódulo si procede
        if self.emphasize_nodule and self.nodule_gain > 0.0:
            factor = 1.0 + self.nodule_gain * nod
            img_masked = np.clip(img_masked * factor, 0.0, 1.0)

        # Sanity check: todas las formas deben coincidir
        # assert img_masked.shape == lung.shape == nod.shape, \
        #     f"Shape mismatch tras bbox para {pid}: img {img_masked.shape}, lung {lung.shape}, nod {nod.shape}"

        vol = img_masked[None, ...].astype(np.float32)  # (1, D, H, W)
        if self.transform:
            vol = self.transform(vol)

        y = torch.tensor(self.labels[pid], dtype=torch.long)
        return vol, y



#transforms
transform = Compose([EnsureType()])

#modelo
def build_model(dropout_prob):
    return DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        dropout_prob=dropout_prob
    )

# train
def train_model_with_internal_validation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path,
    patience=8,          # nº de épocas sin mejorar AUC antes de parar
    min_delta=1e-4       # mejora mínima en AUC para resetear la paciencia
):
    best_val_gmean = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        #train
        model.train()
        running_loss = - 1.0
        all_preds, all_labels = [], []

        print(f"\n Epoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)
        print(f" Entrenamiento — Loss: {running_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | G-Mean: {gmean:.4f}")

        #valid
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels_batch in val_loader:
                val_inputs, val_labels_batch = val_inputs.to(device), val_labels_batch.to(device)
                val_outputs = model(val_inputs)
                batch_loss = criterion(val_outputs, val_labels_batch)
                val_loss += batch_loss.item()

        # métricas completas
        val_acc, val_f1, val_tpr, val_tnr, val_gmean, val_auc = evaluar_modelo(model, val_loader, device)
        print(
            f" Validación   — Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
            f"G-Mean: {val_gmean:.4f} | AUC: {val_auc:.4f}"
        )

        #checkpoint and earlystop
        improved = (not np.isnan(val_gmean)) and (val_gmean > best_val_gmean + min_delta)
        if epoch == 0 or improved:
            best_val_gmean = float(val_gmean)
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f" Guardado modelo con mejor GMEAN ({val_gmean:.4f}) en '{save_path}'")
        else:
            epochs_no_improve += 1
            print(f" Sin mejora de GMEAN ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f" Early stopping activado: {patience} épocas sin mejora de GMEAN ≥ {min_delta}.")
            break


#metricas
def calcular_metricas_binarias(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0, 0, 0
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) else 0
    tnr = TN / (TN + FP) if (TN + FP) else 0
    gmean = np.sqrt(tpr * tnr)
    return tpr, tnr, gmean

def evaluar_modelo(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)                      # (N,2)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob clase positiva
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)

    # AUC puede fallar si solo hay una clase en y_true
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return acc, f1, tpr, tnr, gmean, auc


#cv
def cross_validate(
    model_class,
    dataset,
    batch_size,
    learning_rate,
    preprocessed_dir,
    k=5,
    device='cuda',
    epochs=10,
    save_path_prefix="modelo_densenet_param_1_mask_reforzando_nodulo_early_stop_", 
    weight_decay=1e-5,
    dropout_prob=0.4, 
    seed=42, 
    nodule_gain=None,
):
    results = []
    labels = [dataset.labels[pid] for pid in dataset.patients]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(dataset.patients, labels)):
        print("\n===========================================")
        print(f" Fold {fold+1}/{k} | BS={batch_size} | LR={learning_rate}")
        print("===========================================")

        train_val_subset = Subset(dataset, train_val_idx)
        test_subset = Subset(dataset, test_idx)

        # Internal split 80/20
        internal_labels = [dataset.labels[dataset.patients[i]] for i in train_val_idx]
        train_indices, val_indices = train_test_split(
            train_val_idx,
            test_size=0.2,
            stratify=internal_labels,
            random_state=seed
        )
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = model_class(dropout_prob).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        save_path = f"{save_path_prefix}_bs{batch_size}_lr{learning_rate}_fold{fold+1}.pth"

        # train con validacion interna
        train_model_with_internal_validation(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs,
            save_path, 
            patience=10, 
            min_delta=1e-4
        )

        # evaluacion final en validacion (mejor modelo)
        model.load_state_dict(torch.load(save_path))
        val_acc, val_f1, val_tpr, val_tnr, val_gmean, val_auc = evaluar_modelo(model, val_loader, device)
        test_acc, test_f1, test_tpr, test_tnr, test_gmean, test_auc = evaluar_modelo(model, test_loader, device)


        # guardamos incremental por fold
        row_val = {
            "fold": fold + 1, "set": "VALIDATION",
            "accuracy": val_acc, "f1": val_f1, "tpr": val_tpr, "tnr": val_tnr,
            "gmean": val_gmean, "auc": val_auc,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "dropout_prob": dropout_prob, "seed": seed,
            "preprocessed_dir": preprocessed_dir, 
            "nodule_gain": nodule_gain,
        }
        row_test = {
            "fold": fold + 1, "set": "TEST",
            "accuracy": test_acc, "f1": test_f1, "tpr": test_tpr, "tnr": test_tnr,
            "gmean": test_gmean, "auc": test_auc,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "dropout_prob": dropout_prob, "seed": seed,
            "preprocessed_dir": preprocessed_dir, 
            "nodule_gain": nodule_gain,
        }


        # guardamos las dos filas
        append_rows_to_csv([row_val, row_test])

        #para luego calcular las medias
        results.extend([row_val, row_test])


        # visualizamos gradcam
        gradcam_output_dir = (
            f"gradcam_outputs_param_1_mask_reforzando_nodulo_early_stop_/"
            f"{preprocessed_dir}/fold_{fold+1}_bs{batch_size}_lr{learning_rate}_ng{nodule_gain}_seed{seed}"
            )
        visualizar_gradcams(model, dataset, test_idx, device, gradcam_output_dir)

    results_df = pd.DataFrame(results)

    # añadimos medias
    for split in ["VALIDATION", "TEST"]:
        means = results_df[results_df["set"] == split][["accuracy","f1","tpr","tnr","gmean","auc"]].mean()
        mean_row = {
            "fold": "MEAN", "set": split,
            "accuracy": means["accuracy"], "f1": means["f1"],
            "tpr": means["tpr"], "tnr": means["tnr"],
            "gmean": means["gmean"], "auc": means["auc"],
            "batch_size": batch_size, "learning_rate": learning_rate,
            "seed": seed
        }
        append_rows_to_csv([mean_row])
        results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)


    return results_df

#run grid
preprocessing_dirs = [
    "resize_small_hu_m300_1400_separadas",
    "resize_small_hu_m600_1500_separadas",
    # "resize_medium_hu_m300_1400_separadas", #son demasiado grandes, da out of memory
    # "resize_medium_hu_m600_1500_separadas"
    "resize_cube64_hu_m600_1500", 
    "resize_cube64_hu_m300_1400", 
    "resize_cube128_hu_m600_1500",
    "resize_cube128_hu_m300_1400"
]


learning_rates_to_try = [1e-3]
weight_decays_to_try = [0, 1e-5]
dropout_probs_to_try = [0.3, 0.5]
seeds_to_try = [8]
nodule_gains = [0.2, 0.3, 0.4, 0.5, 0.6]

#batch_sizes_to_try = [4, 8] #16 y 32 dan out of memory en small y medium

def bs_list_for(prep: str):
    # small:  solo 4 y 8 (si no out of memory)
    if "resize_small_" in prep:
        return [4, 8]
    # cubos 64/128: permitir también 16 y 32
    if "resize_cube64" in prep or "resize_cube128" in prep:
        return [4, 8, 16, 32]
    # fallback para otros (ej medium)
    return [4]


all_results = []

for seed in seeds_to_try:
    print("\n##################################################")
    print(f"=== EMPEZANDO EXPERIMENTOS CON SEED: {seed} ===")
    print("##################################################")

    for prep in preprocessing_dirs:
        print("\n##################################################")
        print(f"=== PREPROCESADO: {prep} | SEED={seed} ===")
        print("##################################################")

        base_dir = f"/mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/{prep}/npy"

        for ng in nodule_gains: 
            dataset = LungCTMaskedNPYDataset(
                root_npy_dir=base_dir,
                labels_dict=labels_dict_numeric,
                transform=transform,
                emphasize_nodule=True,
                nodule_gain=ng
            )

            for bs in bs_list_for(prep):
                for lr in learning_rates_to_try:
                    for weight_decay in weight_decays_to_try:
                        for dropout_prob in dropout_probs_to_try:

                            print("\n##################################################")
                            print(f"SEED={seed} | Preprocesado={prep} | nodule_gain={ng} | "
                                  f"BS={bs}, LR={lr}, WD={weight_decay}, Dropout={dropout_prob}")
                            print("##################################################")

                            save_prefix = (
                                f"modelo_densenet_param_1_mask_reforzando_nodulo_early_stop_"
                                f"{prep}_ng{ng}_bs{bs}_lr{lr}_wd{weight_decay}_drop{dropout_prob}_seed{seed}"
                            )

                            df_result = cross_validate(
                                model_class=build_model,
                                dataset=dataset,
                                batch_size=bs,
                                learning_rate=lr,
                                preprocessed_dir=prep,
                                k=5,
                                device=device,
                                epochs=EPOCHS,
                                save_path_prefix=save_prefix,
                                weight_decay=weight_decay,
                                dropout_prob=dropout_prob,
                                seed=seed,
                                nodule_gain=ng   
                            )

                            df_result["preprocessed_dir"] = prep
                            df_result["weight_decay"] = weight_decay
                            df_result["dropout_prob"] = dropout_prob
                            df_result["seed"] = seed
                            df_result["nodule_gain"] = ng
                            all_results.append(df_result)


final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv("resultados_modelo_densenet_param_1_mask_reforzando_nodulo_early_stop.csv", index=False)
print("\n Todos los resultados guardados en 'resultados_modelo_densenet_param_1_mask_reforzando_nodulo_early_stop.csv'")
