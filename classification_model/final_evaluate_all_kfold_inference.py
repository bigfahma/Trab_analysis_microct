import os, json, re, torch, numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import KFold
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from natsort import natsorted
import matplotlib.pyplot as plt

SEED = 42
NUM_FOLDS = 5
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DATA = os.path.join(ROOT_DIR, "Classified_Slices_processed_boneseparated_withoutfibula", "majority_annotation")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "kfold_checkpointsv2")
PROBS_OUT_DIR = os.path.join(ROOT_DIR, "slice_probs_kfold")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PROBS_OUT_DIR, exist_ok=True)

class AutoCrop:
    def __call__(self, img):
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else img

transform = transforms.Compose([
    AutoCrop(),
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

# Step 1: Build list of bones
bones = [(ds, grp, bone) for ds in os.listdir(BASE_DATA)
         for grp in os.listdir(os.path.join(BASE_DATA, ds))
         for bone in os.listdir(os.path.join(BASE_DATA, ds, grp))]

# Step 2: KFold split (same as training)
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(bones))

# Step 3: Loop over each fold
for fold, (_, te_idx) in enumerate(folds):
    print(f"\n[Fold {fold}] Evaluating on test set...")

    # Load trained model
    model = models.resnet18()
    orig = model.conv1
    model.conv1 = torch.nn.Conv2d(1, orig.out_channels, orig.kernel_size,
                                  orig.stride, orig.padding, bias=(orig.bias is not None))
    model.conv1.weight.data = orig.weight.data.mean(dim=1, keepdim=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"best_fold{fold}.pth"), map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Step 4: Loop through each test bone
    for ds, grp, bone in tqdm([bones[i] for i in te_idx], desc=f"Fold {fold} bones"):
        bone_probs = []
        slice_ids = []
        bone_logits = []
        # Collect all image paths in the correct order
        image_paths = []
        bone_dir = os.path.join(BASE_DATA, ds, grp, bone)
        for cls in ['Secondary spongiosa', 'Primary spongiosa', 'Growth plate', 'Epiphyseal bone']:
            cls_dir = os.path.join(bone_dir, cls)
            #print(cls_dir)
            # if not os.path.isdir(cls_dir): 
            #     print('Warning: Directory does not exist:', cls_dir)
            #     continue
            for img in os.listdir(cls_dir):
                if img.endswith(".bmp"):
                    image_paths.append(os.path.join(cls_dir, img))
        
        print(f"Processing {len(image_paths)} slices for {ds}/{grp}/{bone}...")
        for path in image_paths:
            img = Image.open(path).convert("L")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                logits = output.cpu().numpy().squeeze()
                probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
                bone_logits.append(logits.tolist())
            bone_probs.append(probs.tolist())
            slice_ids.append(os.path.basename(path))

        # Save per-bone probabilities
        save_path = os.path.join(PROBS_OUT_DIR, f"fold{fold}_{ds}_{grp}_{bone}_sliceprobs.json")
        # Save complete metadata and inference outputs
        output_json = {
            "dataset": ds,
            "group": grp,
            "bone": bone,
            "fold": fold,
            "logits": bone_logits,
            "slice_ids": slice_ids,
            "probabilities": bone_probs,  # list of [P_class0, P_class1, P_class2, P_class3]
            "class_names": ["Epiphyseal bone","Growth plate","Primary spongiosa","Secondary spongiosa"],
            "n_slices": len(slice_ids),
        }
        #print(output_json)
        # Sanity check - Plot bone probabilities along the Z-axis
        prob_array = np.array(bone_probs)
        # Plot probabilities for each class
        # plt.figure(figsize=(10, 6))
        # for class_idx, class_name in enumerate(output_json["class_names"]):
        #     plt.plot(range(len(prob_array)), prob_array[:, class_idx], label=class_name)

        # plt.title(f"Bone Class Probabilities along Z-axis for {ds}/{grp}/{bone} (Fold {fold})")
        # plt.xlabel("Slice Index (Z-axis)")
        # plt.ylabel("Probability")
        # plt.legend()
        # plt.grid(True)
        # # Save plot as image file
        # plot_save_path = os.path.join(PROBS_OUT_DIR, f"fold{fold}_{ds}_{grp}_{bone}_probabilities_plot.png")
        # #plt.savefig(plot_save_path)
        # #plt.close()
        # plt.show()
        # print(f"Saved plot for probabilities at {plot_save_path}")

        with open(save_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f"Saved probabilities for {ds}/{grp}/{bone} to {save_path}")
print("\n Done: All per-bone slice-wise probabilities saved.")
