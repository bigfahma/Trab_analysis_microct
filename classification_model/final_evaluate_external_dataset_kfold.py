import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from scipy.ndimage import median_filter
from collections import defaultdict

# -------------------- CONFIG --------------------
CHECKPOINT_DIR = "kfold_checkpointsv2"
EXTERNAL_DIR = "Risedronate_15_nifti_cropped_voi"
EXCEL_PATH = "annotations_landmarks.xlsx"
NUM_FOLDS = 5
OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, "external_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- LABEL MAPPING --------------------
class_order = ['Epiphyseal bone', 'Growth plate', 'Primary spongiosa', 'Secondary spongiosa']
idx_cls = {c: i for i, c in enumerate(class_order)}

# -------------------- SMOOTHING + LANDMARKS --------------------
def smooth(probs, k=10):
    sm = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        sm[:, c] = median_filter(probs[:, c], size=k)
    return sm

def find_Zs(sm):
    z12 = next((i for i,p in enumerate(sm) if p[idx_cls["Primary spongiosa"]] >= p[idx_cls["Secondary spongiosa"]]), None)
    zgs = None
    for i,p in enumerate(sm[::-1]):
        if p[idx_cls["Growth plate"]] >= p[idx_cls["Epiphyseal bone"]]:
            zgs = len(sm)-1-i
            break
    z1g = None
    if z12 is not None and zgs is not None and z12 < zgs:
        for i,p in enumerate(sm[z12:zgs+1], start=z12):
            if p[idx_cls["Growth plate"]] >= p[idx_cls["Primary spongiosa"]]:
                z1g = i
                break
    return z12, z1g, zgs

# -------------------- MODEL --------------------
def get_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    return model.to(DEVICE)
def get_all_npy_files_from_nested_root(root_dir):
    file_entries = []
    for group in os.listdir(root_dir):
        group_dir = os.path.join(root_dir, group)
        if not os.path.isdir(group_dir):
            continue
        for file in os.listdir(group_dir):
            if file.endswith('.npy'):
                full_path = os.path.join(group_dir, file)
                file_entries.append((full_path, file, group))
    return file_entries

# -------------------- INFERENCE + EVALUATION --------------------
gt_df = pd.read_excel(EXCEL_PATH)
metrics_all = []
conf_matrices = []

for fold in range(NUM_FOLDS):
    model = get_model()
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold{fold}.pth")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    landmarks_fold = []
    y_true_all, y_pred_all = [], []

    external_file_entries = get_all_npy_files_from_nested_root(EXTERNAL_DIR)
    print(external_file_entries)

    for full_path, file_name, side in external_file_entries:
        bone_id = file_name.replace('_cropped.npy', '.nii')
        row = gt_df[gt_df["Bone Name"].str.contains(bone_id, case=False)]
        print(bone_id, row)
        if row.empty:
            continue
        row = row.iloc[0]
        treatment = row["Group"].strip()
        num_slices = int(row["Slice_ep_plateau"])

        img = np.load(os.path.join(EXTERNAL_DIR, side, file_name))
        x = torch.tensor(img, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(x), dim=1).cpu().numpy()
        smoothed = smooth(probs, k=10)
        z12, z1g, zgs = find_Zs(smoothed)

        # Landmark storage
        landmarks_fold.append({
            "Fold": fold,
            "Bone": bone_id,
            "Z12": z12,
            "Z1G": z1g,
            "ZGS": zgs,
            "Dataset": "External",
            "Group": treatment,
            "NumSlices": num_slices
        })

        # GT Labels
        gt = np.empty(num_slices, dtype=int)
        eg, gp, ps = int(row["Slice eg"]), int(row["Slice gp"]), int(row["Slice ps"])
        for s in range(num_slices):
            if s < ps:        gt[s] = 3
            elif s < gp:      gt[s] = 2
            elif s < eg:      gt[s] = 1
            else:             gt[s] = 0

        # Pred Labels
        pred = np.empty(num_slices, dtype=int)
        for s in range(num_slices):
            if z12 is None or z1g is None or zgs is None:
                pred[s] = 3
            elif s < z12:         pred[s] = 3
            elif s < z1g:         pred[s] = 2
            elif s < zgs:         pred[s] = 1
            else:                 pred[s] = 0

        y_true_all.extend(gt)
        y_pred_all.extend(pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2,3])
    np.save(os.path.join(OUTPUT_DIR, f'cm_external_fold{fold}.npy'), cm)
    conf_matrices.append(cm)

    # Metrics
    metrics_all.append({
        "Fold": fold,
        "Precision_macro": precision_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        "Recall_macro": recall_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        "F1_macro": f1_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        "Accuracy": accuracy_score(y_true_all, y_pred_all)
    })

    with open(os.path.join(OUTPUT_DIR, f'landmarks_external_fold{fold}.json'), 'w') as f:
        json.dump(landmarks_fold, f, indent=2)

# Save metrics summary
with open(os.path.join(OUTPUT_DIR, 'metrics_external_summary.json'), 'w') as f:
    json.dump(metrics_all, f, indent=2)

# -------------------- EVALUATION --------------------

# Mean Confusion Matrix
mean_cm = np.mean(conf_matrices, axis=0)
mean_cm_pct = np.divide(mean_cm, mean_cm.sum(axis=1, keepdims=True), where=mean_cm.sum(axis=1, keepdims=True)!=0) * 100
plt.figure(figsize=(8,6))
sns.heatmap(mean_cm_pct, annot=True, fmt='.1f', cmap='Reds', xticklabels=class_order, yticklabels=class_order)
plt.title('Mean Confusion Matrix Across Folds (%)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mean_cm_external_pct.png'))
plt.close()

# Metrics Table
df = pd.DataFrame(metrics_all)
stats = df[['Precision_macro', 'Recall_macro', 'F1_macro', 'Accuracy']].agg(['mean', 'std'])
stats.to_csv(os.path.join(OUTPUT_DIR, 'metrics_external_table.csv'))
print('Performance metrics on external dataset (mean ± std):')
print(stats)

# Per-group matrix (from landmark JSONs)
group_cm = defaultdict(list)
for fold in range(NUM_FOLDS):
    with open(os.path.join(OUTPUT_DIR, f'landmarks_external_fold{fold}.json')) as f:
        landmarks = json.load(f)
    cm = np.load(os.path.join(OUTPUT_DIR, f'cm_external_fold{fold}.npy'))
    for lm in landmarks:
        group_cm[(lm['Dataset'], lm['Group'])].append(cm)

for (ds, grp), cms in group_cm.items():
    mean_gcm = np.mean(cms, axis=0)
    mean_gcm_pct = np.divide(mean_gcm, mean_gcm.sum(axis=1, keepdims=True), where=mean_gcm.sum(axis=1, keepdims=True)!=0)
    plt.figure(figsize=(8,6))
    sns.heatmap(mean_gcm_pct, annot=True, fmt=".2f", cmap='Reds', cbar=False, square=True, annot_kws={"size": 28})
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])
    fname = f'mean_conf_matrix_pct_{ds}_{grp}.svg'.replace(' ', '_')
    plt.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches='tight', dpi=300)
    plt.close()

# Landmark stats
landmark_stats = {'Z12': [], 'Z1G': [], 'ZGS': []}
for fold in range(NUM_FOLDS):
    with open(os.path.join(OUTPUT_DIR, f'landmarks_external_fold{fold}.json')) as f:
        landmarks = json.load(f)
    for lm in landmarks:
        for k in landmark_stats:
            if lm[k] is not None:
                landmark_stats[k].append(lm[k])
for k, vals in landmark_stats.items():
    if vals:
        print(f'{k}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, n={len(vals)}')

print("External dataset evaluation complete. Results saved in:", OUTPUT_DIR)
