import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from natsort import natsorted, ns
plt.rcParams.update({
    'font.family': 'verdana',
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10
})
PROBS_OUT_DIR = "slice_probs_kfold"  # Path where the JSONs are saved

DATASET_ORDER = ["Sugiyama2008_exp1_roi", "Sugiyama2012_roi", "old21m_roi"]

classes = ["Epiphyseal bone","Growth plate","Primary spongiosa","Secondary spongiosa"]
idx_cls = {c:i for i,c in enumerate(classes)}
def load_grouped_data():
    grouped_data = {}
    for group in os.listdir(PROBS_OUT_DIR):
        group_path = os.path.join(PROBS_OUT_DIR, group)
        if not os.path.isdir(group_path):
            continue
        for file_name in os.listdir(group_path):
            if file_name.endswith("_sliceprobs.json"):
                file_path = os.path.join(group_path, file_name)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        bone = data.get("bone")
                        if not bone:
                            print(f"Missing bone name in file {file_name}, skipping.")
                            continue
                        dataset = "Delisser2024"  # Custom label for your new dataset
                        if dataset not in grouped_data:
                            grouped_data[dataset] = {}
                        if group not in grouped_data[dataset]:
                            grouped_data[dataset][group] = []
                        grouped_data[dataset][group].append(data)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {file_name}, skipping.")
    return grouped_data

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

def smooth(probs, k=10):
    sm = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        sm[:,c] = median_filter(probs[:,c], size=k)
    return sm

def rescale_probabilities(slice_probs, target_length):
    num_slices, num_classes = slice_probs.shape
    # Normalize the slice indices to the target length
    x_old = np.linspace(0, 1, num_slices)
    x_new = np.linspace(0, 1, target_length)
    
    # Interpolate probabilities to the new target length
    rescaled_probs = np.zeros((target_length, num_classes))
    for c in range(num_classes):
        interp_func = interp1d(x_old, slice_probs[:, c], kind='linear', fill_value="extrapolate")
        rescaled_probs[:, c] = interp_func(x_new)
    
    return rescaled_probs

# Function to plot logits for each bone
def plot_logit_for_each_bone(grouped_data):
    landmarks = []
    for dataset, groups in grouped_data.items():
        print(dataset)
        for group, bones in groups.items():
            print(f"  {group} ({len(bones)} bones)")
            max_slices = 0
            for bone_data in bones:
                slice_probs = np.array(bone_data['probabilities'])  # Should be [num_slices, num_classes]
                max_slices = max(max_slices, slice_probs.shape[0])  # Update max slices

            # Initialize list to store aggregated probabilities
            aggregated_probs = []

            for bone_data in natsorted(bones):
               # Extract bone name and probabilities
                bone = bone_data['bone']
                print(f"    {bone} ({len(bone_data['probabilities'])} slices)")
                slice_probs = np.array(bone_data['probabilities'])  # Should be [num_slices, num_classes]
                slice_probs = smooth(slice_probs, k=10)
                slice_probs_rescaled = rescale_probabilities(slice_probs, max_slices)  # Rescale to max slices
                z12, z1g, zgs = find_Zs(slice_probs)
                
                # Store landmarks (Z12, Z1G, ZGS) and additional bone data
                landmarks.append({
                    "Dataset": dataset, 
                    "Group": group, 
                    "Bone Name": bone,
                    "TR-21 (Z12)": int(z12) if z12 is not None else None,
                    "TR-1G (Z1G)": int(z1g) if z1g is not None else None,
                    "TR-GS (ZGS)": int(zgs) if zgs is not None else None,
                    "NumSlices": slice_probs.shape[0]
                })
                print(f"→ {bone} Z12={z12}, Z1G={z1g}, ZGS={zgs}")

                aggregated_probs.append(slice_probs_rescaled)
            
            # Convert the list to a NumPy array for easier manipulation
            aggregated_probs = np.array(aggregated_probs)

            # Calculate mean and standard deviation across all bones for each slice
            mean_probs = np.mean(aggregated_probs, axis=0)  # Mean across bones
            std_probs = np.std(aggregated_probs, axis=0)    # Standard deviation across bones

            # Plot the mean ± std for each class

            print(bone_data["class_names"])
            for class_idx, class_name in enumerate(bone_data["class_names"]):
                ## EP to SS
                mean_curve = mean_probs[::-1, class_idx]
                std_curve = std_probs[::-1, class_idx]

                lower_bound = np.clip(mean_curve - std_curve, 0, 1)
                upper_bound = np.clip(mean_curve + std_curve, 0, 1)

                z = np.arange(len(mean_curve))  # Or use real distances if available

                plt.plot(z, mean_curve, label=class_name)
                plt.fill_between(z, lower_bound, upper_bound, alpha=0.2)
                # # SS to EP
                # lower_bound = np.clip(mean_probs[:, class_idx] - std_probs[:, class_idx], 0, 1)
                # upper_bound = np.clip(mean_probs[:, class_idx] + std_probs[:, class_idx], 0, 1)


                # # Plot
                # z = np.arange(len(mean_probs))  # Or your actual Z-axis in mm if available
                # plt.plot(z, mean_probs[:, class_idx], label=class_name)
                # plt.fill_between(z, lower_bound, upper_bound, alpha=0.2)
            #plt.title(f"Probs for {group} in {dataset}")
            # Remove tick labels but keep the axes
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # No x-axis ticks
            #plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # No y-axis ticks

            plt.xlabel("Slice Index (Z-axis)")
            plt.ylabel("Probability")
            #plt.legend()
            plt.show()

            # Save the plot
            plot_save_path = os.path.join(PROBS_OUT_DIR, f"{dataset}_{group}_logits_plot.svg")
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved logits plot for {dataset}/{group} to {plot_save_path}")
    

    # Save the landmarks to a CSV file
    landmarks_df = pd.DataFrame(landmarks)
    # Sort landmarks by dataset (custom order), group and bone name (natsort)
    landmarks_df["Dataset"] = pd.Categorical(landmarks_df["Dataset"], categories=DATASET_ORDER, ordered=True)

    # Apply the sort per dataset
    sorted_df_list = []
    for dataset in DATASET_ORDER:
        subset = landmarks_df[landmarks_df["Dataset"] == dataset]
        sorted_subset = hierarchical_sort(subset)
        sorted_df_list.append(sorted_subset)

    # Concatenate all sorted subsets
    landmarks_df = pd.concat(sorted_df_list, ignore_index=True)

    landmarks_csv_path = os.path.join(PROBS_OUT_DIR, "landmarks.csv")
    landmarks_df.to_csv(landmarks_csv_path, index=False)
    print(f"\n Landmarks saved to {landmarks_csv_path}")


def hierarchical_sort(df):
    df = df.copy()
    df["Group"] = pd.Categorical(df["Group"], categories=natsorted(df["Group"].unique(), alg=ns.IGNORECASE))
    df["Bone Name"] = pd.Categorical(df["Bone Name"], categories=natsorted(df["Bone Name"].unique(), alg=ns.IGNORECASE))
    return df.sort_values(["Group", "Bone Name"])

# Function to load the saved JSON and group data
def load_grouped_data():
    grouped_data = {}
    for file_name in os.listdir(PROBS_OUT_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(PROBS_OUT_DIR, file_name)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    dataset = data.get("dataset")
                    group = data.get("group")
                    bone = data.get("bone")

                    if not all([dataset, group, bone]):
                        print(f"Missing keys in file {file_name}, skipping.")
                        continue

                    if dataset not in grouped_data:
                        grouped_data[dataset] = {}
                    if group not in grouped_data[dataset]:
                        grouped_data[dataset][group] = []

                    grouped_data[dataset][group].append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file_name}, skipping.")
    return grouped_data

# Load and plot
grouped_data = load_grouped_data()
plot_logit_for_each_bone(grouped_data)

print("\n All plots generated successfully!")
