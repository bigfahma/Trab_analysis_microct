 # final_evaluate_all_kfold.py
"""
Script to fully evaluate k-fold classification results for scientific reporting.
- Plots train/val loss curves for all folds
- Computes and displays mean/std for precision, recall, F1, accuracy
- Plots mean confusion matrix
- Optionally summarizes landmark statistics
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CHECKPOINT_DIR = "kfold_checkpointsv2"
NUM_FOLDS = 5  # Change if needed

# --- Load losses ---
with open(os.path.join(CHECKPOINT_DIR, 'losses.json'), 'r') as f:
    losses = json.load(f)
train_losses = losses['train']
val_losses = losses['val']

# --- Plot loss curves ---
fig, axes = plt.subplots(NUM_FOLDS, 1, figsize=(10, 3 * NUM_FOLDS))
for i, (tr, va) in enumerate(zip(train_losses, val_losses)):
    ax = axes[i] if NUM_FOLDS > 1 else axes
    ax.plot(range(1, len(tr)+1), tr, label='Train')
    ax.plot(range(1, len(va)+1), va, label='Val', linestyle='--')
    ax.set_title(f'Fold {i} Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel(f'Epoch (1-{max(len(tr), len(va))})')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curves_subplots.png'))
plt.show()

# --- Load metrics and confusion matrices ---
conf_matrices = []
for fold in range(NUM_FOLDS):
    lm_path = os.path.join(CHECKPOINT_DIR, f'landmarks_fold{fold}.json')
    with open(lm_path, 'r') as f:
        landmarks = json.load(f)
    cm_path = os.path.join(CHECKPOINT_DIR, f'cm_fold{fold}.npy')
    cm = np.load(cm_path)
    conf_matrices.append(cm)

# --- Load metrics from metrics_summary.json (created by training script) ---
metrics_path = os.path.join(CHECKPOINT_DIR, 'metrics_summary.json')
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
else:
    print('metrics_summary.json not found. Please ensure the training script saved it.')

# --- Load metrics from summary (if available) ---
# If you saved metrics_summary in a file, load it here
summary_path = os.path.join(CHECKPOINT_DIR, 'metrics_summary.json')
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        metrics = json.load(f)
else:
    print('metrics_summary.json not found. Please add metrics to this file for full table.')


# --- Compute mean confusion matrix as percentage ---
mean_cm = np.mean(conf_matrices, axis=0)
row_sums = mean_cm.sum(axis=1, keepdims=True)
mean_cm_pct = np.divide(mean_cm, row_sums, where=row_sums!=0) * 100
plt.figure(figsize=(8, 6))
sns.heatmap(mean_cm_pct, annot=True, fmt='.1f', cmap='Reds')
plt.title('Mean Confusion Matrix Across Folds (%)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'mean_conf_matrix_pct.png'))
plt.show()

# --- Per-group, per-dataset confusion matrices ---
from collections import defaultdict
group_cm = defaultdict(list)  # {(dataset, group): [cm]}
for fold in range(NUM_FOLDS):
    lm_path = os.path.join(CHECKPOINT_DIR, f'landmarks_fold{fold}.json')
    with open(lm_path, 'r') as f:
        landmarks = json.load(f)
    cm_path = os.path.join(CHECKPOINT_DIR, f'cm_fold{fold}.npy')
    cm = np.load(cm_path)
    # Try to group by dataset/group from landmarks
    for lm in landmarks:
        ds = lm.get('Dataset')
        grp = lm.get('Group')
        if ds is not None and grp is not None:
            group_cm[(ds, grp)].append(cm)

for (ds, grp), cms in group_cm.items():
    mean_gcm = np.mean(cms, axis=0)
    row_sums = mean_gcm.sum(axis=1, keepdims=True)
    mean_gcm_pct = np.divide(mean_gcm, row_sums, where=row_sums!=0) 
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_gcm_pct, annot=True, fmt='.2f', cmap='Reds', cbar = False, square = True, annot_kws={"size": 28} )        
    #plt.title(f'Mean Confusion Matrix: {ds} / {grp} (%)')
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    # Remove the ticks from the two axes
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])

    # Remove the heatmap bar
    plt.tight_layout()
    fname = f'mean_conf_matrix_pct_{ds}_{grp}.svg'.replace(' ', '_')
    plt.savefig(os.path.join(CHECKPOINT_DIR, fname), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

# --- Metrics Table ---
if metrics:
    df = pd.DataFrame(metrics)
    # Use correct column names from metrics_summary.json
    stats = df[['Precision_macro', 'Recall_macro', 'F1_macro', 'Accuracy']].agg(['mean', 'std'])
    print('Performance metrics (mean ± std):')
    print(stats)
    stats.to_csv(os.path.join(CHECKPOINT_DIR, 'metrics_table.csv'))
else:
    print('No metrics loaded. Please ensure metrics_summary.json is available.')

# --- Optionally: summarize landmark statistics ---
# Example: mean Z12, Z1G, ZGS per fold
landmark_stats = {'Z12': [], 'Z1G': [], 'ZGS': []}
for fold in range(NUM_FOLDS):
    lm_path = os.path.join(CHECKPOINT_DIR, f'landmarks_fold{fold}.json')
    with open(lm_path, 'r') as f:
        landmarks = json.load(f)
    for lm in landmarks:
        for k in ['Z12', 'Z1G', 'ZGS']:
            if lm[k] is not None:
                landmark_stats[k].append(lm[k])
                print(f'Fold {fold} - {k}: {lm[k]}')
for k, vals in landmark_stats.items():
    if vals:
        print(f'{k}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, n={len(vals)}')

print('Evaluation complete. Figures and tables saved to', CHECKPOINT_DIR)

# Define class labels for consistency
class_labels = ['Epiphyseal bone', 'Growth plate', 'Primary spongiosa', 'Secondary spongiosa']
short_labels = ['Epiphyseal', 'Growth', 'Primary', 'Secondary']

latex_lines = []
latex_lines.append(r"\begin{table}[!ht]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Performance evaluation of the classification model for classifying 2D micro-CT slices of the proximal mouse tibia into four compartments: epiphyseal bone, growth plate, primary spongiosa, and secondary spongiosa. Results are shown across all groups in Dataset 1 \cite{sugiyama2008mechanical}, Dataset 2 \cite{sugiyama2012bones}, Dataset 3 \cite{meakin2015disuse}, and the external dataset \cite{sugiyama2011risedronate}, using Precision (Prec), Recall (Rec), and F1-score (F1). PTH0, PTH20, PTH40, and PTH80 denote parathyroid hormone doses of 0 to 80 μg/kg/day. ML denotes mechanical loading.}")
latex_lines.append(r"\small")
latex_lines.append(r"\begin{tabular}{lllcccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r"\textbf{Dataset} & \textbf{Treatment} & \textbf{Metric} & \textbf{Epiphyseal} & \textbf{Growth} & \textbf{Primary} & \textbf{Secondary} \\")
latex_lines.append(r"& & & \textbf{bone} & \textbf{plate} & \textbf{spongiosa} & \textbf{spongiosa} \\")
latex_lines.append(r"\midrule")

from collections import defaultdict

# Format function with ±
def format_mean_std(values):
    return " & ".join(f"${np.mean(v):.3f} \\pm {np.std(v):.3f}$" for v in zip(*values))

for (ds, grp), cms in sorted(group_cm.items()):
    # Initialize lists for each metric per class
    precision_all, recall_all, f1_all = [], [], []

    for cm in cms:
        cm_int = np.round(cm).astype(int)
        TP = np.diag(cm_int)
        with np.errstate(divide='ignore', invalid='ignore'):
            prec = TP / cm_int.sum(axis=0)
            rec  = TP / cm_int.sum(axis=1)
            f1   = 2 * (prec * rec) / (prec + rec)
        precision_all.append(np.nan_to_num(prec))
        recall_all.append(np.nan_to_num(rec))
        f1_all.append(np.nan_to_num(f1))

    # Add LaTeX lines (mean ± std per class)
    latex_lines.append(r"\multirow{3}{*}{" + ds + "} & \multirow{3}{*}{" + grp + "} & Prec & " +
                       format_mean_std(precision_all) + r" \\")
    latex_lines.append(r"& & Rec  & " + format_mean_std(recall_all) + r" \\")
    latex_lines.append(r"& & F1   & " + format_mean_std(f1_all) + r" \\")

    latex_lines.append(r"\midrule")


latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\label{tab:1classification-performance}")
latex_lines.append(r"\end{table}")

# Save to file
latex_str = "\n".join(latex_lines)
with open(os.path.join(CHECKPOINT_DIR, "performance_table.tex"), "w", encoding="utf-8") as f:

    f.write(latex_str)

print("LaTeX performance table saved to:", os.path.join(CHECKPOINT_DIR, "performance_table.tex"))
