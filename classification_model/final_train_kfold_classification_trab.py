import os, json, re, torch, numpy as np
import random
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from scipy.ndimage import median_filter

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ————————— SANITY CHECK CONFIG —————————
SANITY_CHECK = False   # Set to False for full run
SANITY_FRACTION = 0.01
if SANITY_CHECK:
    NUM_FOLDS = 2
    NUM_EPOCHS = 2
    PATIENCE = 1
    CKPT_INTERVAL = 1
    BATCH_SIZE = 2
else:
    NUM_FOLDS = 5
    NUM_EPOCHS = 100
    PATIENCE = 15
    CKPT_INTERVAL = 3
    BATCH_SIZE = 64

BASE_DATA = "Classified_Slices_processed_boneseparated_withoutfibula/majority_annotation" #Directory with the datasets
CHECKPOINT_DIR = "kfold_checkpointsv2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ', DEVICE)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ————————— DATASET & DATALOADER —————————
class AutoCrop:
    def __call__(self, img):
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else img
class ClassifiedDataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(e['image_path']).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, e['label'], e['image_path']  # Return path as well
    

def build_image_entries(bones):
    entries = []
    for ds, grp, bone in bones:
        bone_dir = os.path.join(BASE_DATA, ds, grp, bone)
        for cls in os.listdir(bone_dir):
            cls_dir = os.path.join(bone_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = ["Epiphyseal bone","Growth plate","Primary spongiosa","Secondary spongiosa"].index(cls)
            for img in os.listdir(cls_dir):
                if img.endswith('.bmp'):
                    entries.append({
                        'image_path': os.path.join(cls_dir, img),
                        'label': label
                    })
    return entries

def build_loaders(train_bones, test_bones):
    train_entries = build_image_entries(train_bones)
    test_entries = build_image_entries(test_bones)

    # --- SANITY CHECK: Use only a fraction of the data ---
    if SANITY_CHECK:
        n_train = int(len(train_entries) * SANITY_FRACTION)
        n_test = int(len(test_entries) * SANITY_FRACTION)
        train_entries = train_entries[:max(1, n_train)]
        test_entries = test_entries[:max(1, n_test)]

    tf_train = transforms.Compose([AutoCrop(), transforms.Resize((384,384)),
                                   transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=0.2),
                                   transforms.ToTensor()])
    tf_test = transforms.Compose([AutoCrop(), transforms.Resize((384,384)), transforms.ToTensor()])
    train_ds = ClassifiedDataset(train_entries, tf_train)
    test_ds = ClassifiedDataset(test_entries, tf_test)
    return {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True),
        'val':   DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    }, {'train': len(train_ds), 'val': len(test_ds)}

# ————————— MODEL SETUP —————————
model = models.resnet18(pretrained=True)
orig = model.conv1
model.conv1 = torch.nn.Conv2d(1, orig.out_channels, orig.kernel_size, orig.stride, orig.padding, bias=(orig.bias is not None))
model.conv1.weight.data = orig.weight.data.mean(dim=1, keepdim=True)
for name, param in model.named_parameters():
    param.requires_grad = 'fc' in name
model.fc = torch.nn.Linear(model.fc.in_features, 4)
device = DEVICE
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# ————————— SLICE & LANDMARK HELPERS —————————
slice_re = re.compile(r"_slice_(\d+)")
def slice_idx(path):
    m = slice_re.search(os.path.basename(path))
    return int(m.group(1)) if m else -1

classes = ["Epiphyseal bone","Growth plate","Primary spongiosa","Secondary spongiosa"]
idx_cls = {c:i for i,c in enumerate(classes)}

def smooth(probs, k=15):
    sm = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        sm[:,c] = median_filter(probs[:,c], size=k)
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
    # --- DEBUG: Print transitions in sanity check ---
    if SANITY_CHECK:
        print(f"  [DEBUG] Z12={z12}, Z1G={z1g}, ZGS={zgs}")
        print(f"  [DEBUG] sm shape: {sm.shape}")
        print(f"  [DEBUG] sm (first 5 rows):\n{sm[:5]}")
    return z12, z1g, zgs

# ————————— K-FOLD EXECUTION —————————
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
bones = [(ds,grp,bone) for ds in os.listdir(BASE_DATA)
         for grp in os.listdir(os.path.join(BASE_DATA, ds))
         for bone in os.listdir(os.path.join(BASE_DATA, ds, grp))]
folds = list(kf.split(bones))

metrics_summary = []

# --- For saving losses ---
all_train_losses = []
all_val_losses = []

for fold, (tr_idx, te_idx) in enumerate(folds):
    train_bones = [bones[i] for i in tr_idx]
    test_bones = [bones[i] for i in te_idx]
    loaders, sizes = build_loaders(train_bones, test_bones)

    best_val_loss, wait = float('inf'), 0
    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"Fold {fold}, Epoch {epoch}")
        epoch_train_loss = None
        epoch_val_loss = None
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            running_loss, total = 0.0, 0
            for x,y,_ in tqdm(loaders[phase], desc=f"{phase}"):
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out = model(x)
                    loss = criterion(out, y)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()*x.size(0)
                total += x.size(0)
            avg_loss = running_loss/total
            print(f"  {phase} loss={avg_loss:.4f}")
            if phase=='train':
                epoch_train_loss = avg_loss
            if phase=='val':
                epoch_val_loss = avg_loss
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_fold{fold}.pth"))
                    wait = 0
                else:
                    wait += 1
        # Save losses for this epoch
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        if (epoch+1)%CKPT_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"fold{fold}_ep{epoch+1}.pth"))
        if wait >= PATIENCE:
            print("Early stopping")
            break

    # Save losses for this fold
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"best_fold{fold}.pth")))

    # ——— TEST EVALUATION & EXTRACTIONS —— 
    preds, targets = [], []
    landmarks = []
    bone_probs = {}  # {(ds, grp, bone): [(slice_idx, prob), ...]}
    all_probs = []  # Collect all validation probabilities for ROC/AUC

    for ds, grp, bone in test_bones:
        bone_probs[(ds, grp, bone)] = []

    for x, y, paths in loaders['val']:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
            p = out.argmax(dim=1)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
        preds += p.cpu().tolist()
        targets += y.cpu().tolist()
        all_probs.extend(probs)  # Collect all probabilities
        for i, path in enumerate(paths):
            parts = path.split(os.sep)
            ds, grp, bone = parts[-5], parts[-4], parts[-3]  # Corrected indices
            sidx = slice_idx(path)
            bone_probs[(ds, grp, bone)].append((sidx, probs[i]))
    for (ds, grp, bone), slice_probs in bone_probs.items():
        if not slice_probs:
            continue
        slice_probs.sort(key=lambda x: x[0])
        pr = np.array([prob for _, prob in slice_probs])
        sm = smooth(pr, k=10)
        z12, z1g, zgs = find_Zs(sm)
        landmarks.append({
            "Dataset": ds, "Group": grp, "Bone": bone,
            "Z12": int(z12) if z12 is not None else None,
            "Z1G": int(z1g) if z1g is not None else None,
            "ZGS": int(zgs) if zgs is not None else None,
            "NumSlices": sm.shape[0]
        })
        print(f"→ {bone} Z12={z12}, Z1G={z1g}, ZGS={zgs}")

    # Macro metrics
    prec_macro = precision_score(targets, preds, average='macro')
    rec_macro  = recall_score(targets, preds, average='macro')
    f1_macro   = f1_score(targets, preds, average='macro')
    acc        = accuracy_score(targets, preds)
    cm         = confusion_matrix(targets, preds)

    # Per-class metrics
    prec_perclass = precision_score(targets, preds, average=None)
    rec_perclass  = recall_score(targets, preds, average=None)
    f1_perclass   = f1_score(targets, preds, average=None)

    print(f"Fold {fold} metrics: P={prec_macro:.3f} R={rec_macro:.3f} F1={f1_macro:.3f} A={acc:.3f}")
    print("CM:", cm)

    np.save(os.path.join(CHECKPOINT_DIR, f"cm_fold{fold}.npy"), cm)
    with open(os.path.join(CHECKPOINT_DIR, f"landmarks_fold{fold}.json"), 'w') as fw:
        json.dump(landmarks, fw, indent=2)

    metrics_summary.append({
        "Fold": fold,
        "Precision_macro": prec_macro,
        "Recall_macro": rec_macro,
        "F1_macro": f1_macro,
        "Accuracy": acc,
        "Precision_perclass": prec_perclass.tolist(),
        "Recall_perclass": rec_perclass.tolist(),
        "F1_perclass": f1_perclass.tolist()
    })

    # ROC Curves & AUC
    from sklearn.metrics import roc_auc_score, roc_curve
    roc_data = {}
    try:
        targets_onehot = np.eye(len(classes))[np.array(targets)]
        all_probs_np = np.array(all_probs)
        for i, cls_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(targets_onehot[:, i], all_probs_np[:, i])
            auc = roc_auc_score(targets_onehot[:, i], all_probs_np[:, i])
            roc_data[cls_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}
    except Exception as e:
        print(f"ROC/AUC calculation failed: {e}")
    with open(os.path.join(CHECKPOINT_DIR, f"roc_fold{fold}.json"), "w") as fw:
        json.dump(roc_data, fw, indent=2)

    # Prediction Probability Analysis: save per-slice probabilities for plotting mean±std vs slice index
    prob_stats = {}  # {(ds, grp, bone, cls): {slice_idx: prob}}
    for (ds, grp, bone), slice_probs in bone_probs.items():
        if not slice_probs:
            continue
        # slice_probs: list of (slice_idx, [prob_class0, prob_class1, ...])
        for i, cls_name in enumerate(classes):
            slice_dict = {}
            for sidx, prob_vec in slice_probs:
                slice_dict[int(sidx)] = float(prob_vec[i])
            key = f"{ds}|{grp}|{bone}|{cls_name}"
            prob_stats[key] = slice_dict
    with open(os.path.join(CHECKPOINT_DIR, f"prob_slices_fold{fold}.json"), "w") as fw:
        json.dump(prob_stats, fw, indent=2)

# --- Save all losses for plotting ---
with open(os.path.join(CHECKPOINT_DIR, 'losses.json'), 'w') as f:
    json.dump({'train': all_train_losses, 'val': all_val_losses}, f)

# --- Save metrics summary for evaluation ---
with open(os.path.join(CHECKPOINT_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)

# ————————— SUMMARY —————————
print("\n=== Classification Summary Across Folds ===")
for m in metrics_summary:
    print(f"Fold {m['Fold']}: P={m['Precision_macro']:.3f} R={m['Recall_macro']:.3f} F1={m['F1_macro']:.3f} Acc={m['Accuracy']:.3f}")