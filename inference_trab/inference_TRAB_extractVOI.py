import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from PIL import Image
import numpy as np 
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib.widgets import Slider
import SimpleITK as sitk
import re
from PIL import Image, ImageFile
from scipy.ndimage import median_filter
import traceback
import json

class AutoCrop:
    """
    Auto-crops a grayscale PIL image to remove extra background border around the object.
    If the background is pure white (pixel=255) or pure black (pixel=0), this works well.
    Otherwise, you might adjust the approach or threshold.
    """
    def __init__(self):
        """
        Parameters:
            background (int): The assumed background pixel value. 
                              Use 255 if your background is white, 0 if black.
        """

    def __call__(self, img):
        """
        img: A PIL Image in 'L' mode (grayscale).
        Returns: A cropped PIL Image.
        """
        bbox = img.getbbox()
        if bbox:
            return img.crop(bbox)
        else:
            return img

############################################################
# 4) HELPER FUNCTIONS
############################################################

def extract_slice_idx(filename):
    match = re.search(r"_slice_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1



def smoothen_probabilities(probabilities, kernel_size=10):
    probabilities = np.array(probabilities)
    
    smoothed_probabilities = np.zeros_like(probabilities)
    for i in range(probabilities.shape[1]):
        smoothed_probabilities[:, i] = median_filter(probabilities[:, i], size=kernel_size)
    
    return smoothed_probabilities


def invert_z_axis(image):
    """
    Inverts the slices along the z-axis in a 3D image.

    Parameters:
    image (sitk.Image): The input 3D image.

    Returns:
    sitk.Image: The 3D image with inverted z-axis slices.
    """
    # Create a flip filter
    flip_filter = sitk.FlipImageFilter()
    
    # Set the axis to flip (only the z-axis, which is the third dimension)
    flip_axes = [False, False, True]
    flip_filter.SetFlipAxes([False, False, True])
    
    # Apply the flip filter
    inverted_image = flip_filter.Execute(image)
    
    return inverted_image

def plot_probability_distributions(slice_probs, class_names):
    """
    Plot the predicted class probability distribution along the z-axis.
    
    Args:
        slice_probs (numpy.ndarray): Array of shape (num_slices, num_classes)
        class_names (list): List of class name strings.
    """
    z_axis = np.arange(slice_probs.shape[0])  # Number of slices along the z-axis
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        plt.plot(z_axis, slice_probs[:, i], label=f'{class_name} Probability')
    
    plt.title("Class Probability Distribution Along the Bone Length (z-axis)")
    plt.xlabel("Slice (z-axis)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

def find_Z12(slice_probs, p=0.2):
    """
    Identifies Z12 as the first slice where Primary >= Secondary spongiosa.

    If Primary spongiosa is not confidently detected across the volume
    (mean probability < p), fallback to slice where Growth plate dominates.

    Parameters:
        slice_probs (List[np.array]): Class probabilities per slice.
        p (float): Threshold for determining Primary spongiosa existence.

    Returns:
        int or None: Slice index of Z12 landmark, or fallback Growth slice.
    """
    idx_growth = class_order.index("Growth plate")
    idx_primary = class_order.index("Primary spongiosa")
    idx_secondary = class_order.index("Secondary spongiosa")

    # Check global presence of Primary spongiosa
    max_primary_probs = [probs[idx_primary] for probs in slice_probs]
    if np.max(max_primary_probs) < p:
        # Fallback: first dominant Growth plate slice
        for z, probs in enumerate(slice_probs):
            if probs[idx_growth] > probs[idx_primary] and probs[idx_growth] > probs[idx_secondary]:
                print(f"Fallback Zps at slice {z} due to low primary spongiosa presence")
                return z
        return None

    # Normal logic: Primary >= Secondary
    for z, probs in enumerate(slice_probs):
        if probs[idx_primary] >= probs[idx_secondary]:
            print(z, probs[idx_primary], probs[idx_secondary])
            return z

    return None

def find_ZGS(slice_probs):
    """
    ZGS also from the proximal side (the "back"),
    so we reverse the array and detect 'Epiphyseal >= Growth'.

    The first crossing in reversed => last crossing forward.
    """

    idx_epi = class_order.index("Epiphyseal bone")
    idx_growth = class_order.index("Growth plate")
  
    reversed_probs = slice_probs[::-1]
  
    for i, probs in enumerate(reversed_probs):
        if probs[idx_growth] >= probs[idx_epi]:
            forward_idx = len(slice_probs) - 1 - i
            return forward_idx
    return None

def find_Z1G_in_subrange(slice_probs, start_idx, end_idx, p=0.2):
    """
    Identify the Z1G landmark within a subrange of axial slices.

    Z1G must lie strictly between Z12 (start_idx) and ZGS (end_idx).
    The search is conducted over slice_probs[start_idx:end_idx+1] in forward order,
    looking for the condition: Growth plate probability >= Primary spongiosa probability.

    If the maximum Primary spongiosa probability across the VOI is consistently
    below a threshold p (default 0.2), Z1G is considered non-existent.

    Parameters:
        slice_probs (List[np.array]): Class probabilities per slice, shape (N_slices, N_classes).
        start_idx (int): Index for Z12 landmark (start of search range).
        end_idx (int): Index for ZGS landmark (end of search range).
        p (float): Threshold for excluding Z1G if Primary spongiosa is poorly detected.

    Returns:
        int or None: Index of Z1G if condition met; otherwise None.
    """
    if start_idx >= end_idx:
        return None

    idx_primary = class_order.index("Primary spongiosa")
    idx_growth = class_order.index("Growth plate")

    subarray = slice_probs[start_idx:end_idx+1]

    for i, probs in enumerate(subarray):
        if probs[idx_growth] >= probs[idx_primary]:
            return start_idx + i

    return None

def plot_slices_save_hr(slices, save_path="output.svg", dpi=300):
    """
    Plots three subsections from the provided slices in a single row without any extra elements.
    
    Parameters:
        slices (numpy.ndarray): A NumPy array of shape (num_slices, H, W) containing image slices.
        save_path (str): Path to save the output figure as an .svg file.
        dpi (int): Resolution of the saved figure.

    Returns:
        None
    """
    if slices.shape[0] < 3:
        raise ValueError("At least three slices are required for plotting.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 3 subplots in a row

    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap="bone")  # Display as grayscale image
        ax.axis("off")  # Remove axes

    plt.subplots_adjust(wspace=0.05, hspace=0)  # Small fixed space between subplots
    plt.savefig(save_path, format="svg", dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def load_slices_np_with_preprocessing(bone_folder):
    """
    Load and preprocess slices from the bone volume using the provided transformation.
    
    Returns:
        torch.Tensor: A tensor of shape (num_slices, C, H, W)
    """
    # Read the 3D volume using SimpleITK
    im = sitk.ReadImage(bone_folder)
    slices = []
    num_slices = im.GetSize()[2]
    for slice_idx in range(num_slices):
        # Convert the current slice to a NumPy array and then to a PIL image
        slice_array = sitk.GetArrayFromImage(im[:, :, slice_idx])
        pil_img = Image.fromarray(slice_array)
        # Apply the transformation (resize, tensor conversion, etc.)
        slices.append(pil_img)
    
    del im
    return np.stack(slices)  


def load_slices_with_preprocessing(bone_folder, data_transform):
    """
    Load and preprocess slices from the bone volume using the provided transformation.
    
    Returns:
        torch.Tensor: A tensor of shape (num_slices, C, H, W)
    """
    or_im = sitk.ReadImage(bone_folder)
    or_im = or_im[:,:,int(0.75 * or_im.GetDepth()):] # CROP 75%-100%
    slices = []
    num_slices = or_im.GetSize()[2]
    for slice_idx in range(num_slices):
        slice_array = sitk.GetArrayFromImage(or_im[:, :, slice_idx])
        pil_img = Image.fromarray(slice_array)
        img_tensor = data_transform(pil_img).unsqueeze(0)
        slices.append(img_tensor)
    
    return torch.stack(slices), or_im  


def predict_slice_probabilities(model, slices_tensor, batch_size=1):
    """
    Predict class probabilities for each slice by processing the slices in mini-batches.
    
    Args:
        model: The PyTorch model.
        slices_tensor (torch.Tensor): Tensor containing all slices.
        batch_size (int): Mini-batch size for inference.
    
    Returns:
        numpy.ndarray: Array of probabilities for each slice.
    """
    model.eval()
    # Get the device from the model parameters (assumes model is already on GPU)
    device = next(model.parameters()).device
    probabilities = []
    
    with torch.no_grad():
        for i in range(0, slices_tensor.size(0)):
            batch = slices_tensor[i].to(device)
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probabilities.append(probs.cpu())
            
            del batch, outputs, probs
            torch.cuda.empty_cache()
    
    return torch.cat(probabilities, dim=0).numpy()
    
def visualize_tensor_slices(tensor, cmap='bone', channel=0):
    """
    Visualize slices from a 3D (N, H, W) or 4D (N, C, H, W) tensor with a slider.

    Args:
        tensor (torch.Tensor or np.ndarray): Shape (N, H, W) or (N, C, H, W)
        cmap (str): Colormap (default: 'bone')
        channel (int): Channel to display if tensor has channel dimension
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    print(tensor.shape)
    # Accept (N, C, H, W) or (N, H, W)
    if tensor.ndim == 5:
        slices = tensor[:, 0, channel]
    elif tensor.ndim == 4:
        slices = tensor[:, channel]
    elif tensor.ndim == 3:
        slices = tensor
    else:
        raise ValueError("Tensor must be shape (N, C, H, W) or (N, H, W)")
    
    num_slices = slices.shape[0]
    current_slice = num_slices // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.18)
    img = ax.imshow(slices[current_slice], cmap=cmap)
    ax.set_title(f"Slice {current_slice+1} / {num_slices}")
    ax.axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=current_slice, valstep=1)

    def update(val):
        idx = int(slider.val)
        img.set_data(slices[idx])
        ax.set_title(f"Slice {idx+1} / {num_slices}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def visualize_multiple_tensor_stacks(tensors, titles=None, cmap='bone', channel=0):
    """
    Visualizes multiple stacks of image slices side by side, each with its own slider.

    Args:
        tensors: list of tensors or numpy arrays, each of shape (N,H,W) or (N,C,H,W)
        titles: list of strings, optional, titles for each subplot
        cmap: str, matplotlib colormap
        channel: int, channel to show if (N,C,H,W)
    """
    n = len(tensors)
    tensors_np = []

    for t in tensors:
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        if t.ndim == 4:  # (N, C, H, W)
            t = t[:, channel]
        elif t.ndim != 3:
            raise ValueError("Each tensor must be (N,H,W) or (N,C,H,W)")
        tensors_np.append(t)

    num_slices = [x.shape[0] for x in tensors_np]
    slice_idxs = [k//2 for k in num_slices]

    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    imgs = []
    sliders = []

    plt.subplots_adjust(bottom=0.16)

    for i, ax in enumerate(axes):
        img = ax.imshow(tensors_np[i][slice_idxs[i]], cmap=cmap)
        imgs.append(img)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
        else:
            ax.set_title(f"Stack {i+1}")

        ax_slider = plt.axes([0.13 + i*0.8/n, 0.08, 0.7/n, 0.03])
        slider = Slider(ax_slider, f"Slice {i+1}", 0, num_slices[i]-1, 
                        valinit=slice_idxs[i], valstep=1)
        sliders.append(slider)

    def update(val):
        for i, slider in enumerate(sliders):
            idx = int(slider.val)
            imgs[i].set_data(tensors_np[i][idx])
            axes[i].set_title(
                titles[i] if titles else f"Stack {i+1} (Slice {idx}/{num_slices[i]})"
            )
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    model = models.resnet18()

    # Modify the first convolutional layer to accept 1 channel (grayscale)
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,  
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None,
    )

    # Copy the pre-trained weights from the original layer
    model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
    for name, param in model.named_parameters():
        if "fc" in name:  
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_classes = 4  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(r"final_classification_model.pth")) # Path to the final classification ckpt .pth

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Prevent issues with truncated images

    resolution = 4.78
    data_transform = transforms.Compose([
        AutoCrop(),  
        transforms.Resize((384 , 384)),
        transforms.ToTensor(),
    ])

    class_order = ["Epiphyseal bone", "Growth plate", "Primary spongiosa", "Secondary spongiosa"]

    data_dir = r'2.1.Prealigned' # Input directory with the prealigned bones
    output_dir = r'3.VOI_extraction' # Output directory
    os.makedirs(output_dir, exist_ok=True)
    group_folders = [
        f for f in natsorted(os.listdir(data_dir)) 
        if os.path.isdir(os.path.join(data_dir,f))
    ]

    STOP_METAPHYSEAL = 1 #0.5 # mm 
    STOP_EPIPHYSEAL = 0.25 # mm
    for group_folder in group_folders:

        group_path = os.path.join(data_dir, group_folder)
        print(f"Processing group: {group_folder}")

        bone_folders = [
            f for f in natsorted(os.listdir(group_path)) 
            if  os.path.isdir(os.path.join(group_path, f))
        ]

        for bone_folder in bone_folders:
            try:
                bone_folder_path = os.path.join(group_path, bone_folder)
                bone_filename = [f for f in os.listdir(bone_folder_path) if f.endswith('.nii.gz')][0]
                bone_file_path = os.path.join(bone_folder_path, bone_filename)

                print(f"Image: {bone_filename}")
                output_bone_folder = os.path.join(output_dir, group_folder, bone_folder)
                os.makedirs(output_bone_folder, exist_ok=True)

                print(f"Processing bone folder: {bone_filename}")
                # skip if already processed
                if os.path.exists(os.path.join(output_bone_folder, bone_filename + "_VOI_secondary.nii.gz")):
                    print(f"Skipping {bone_filename} as it has already been processed.")
                    continue
                slices_tensor, im = load_slices_with_preprocessing(bone_file_path, data_transform)
                #visualize_tensor_slices(slices_tensor, cmap='bone')
                #visualize_slices(im)
                slice_probs = predict_slice_probabilities(model, slices_tensor)

                # plot the slice probs
                z = np.arange(len(slice_probs))  # Or use real distances 

                #plt.plot(z, slice_probs)
                #plt.show()
                json_outfile = os.path.join(output_bone_folder, f"{bone_filename}_sliceprobs.json")

                output_json = {
                    "group": group_folder,
                    "bone": bone_filename,
                    "probabilities": slice_probs.tolist(),  # or post-processed
                    "class_names": ["Epiphyseal bone", "Growth plate", "Primary spongiosa", "Secondary spongiosa"],
                }
                with open(json_outfile, "w") as f:
                    json.dump(output_json, f, indent=2)

                # Smooth the probability curves
                slice_probs = smoothen_probabilities(slice_probs, kernel_size=10)
                
                #plot_probability_distributions(slice_probs, class_order)  

                pred_zps = find_Z12(slice_probs) 
                print(f"Predicted Zps: {pred_zps}")
                pred_zeg = find_ZGS(slice_probs)
                print(f"Predicted Zeg: {pred_zeg}")
                pred_zgp = find_Z1G_in_subrange(slice_probs, pred_zps, pred_zeg)
                print(f"Predicted Zgp: {pred_zgp}")
                VOI_secondary = im[:,:,:pred_zps]
                VOI_epiphyseal = im[:,:,pred_zeg:]
                STOP_METAPHYSEAL_VOXELS = int(STOP_METAPHYSEAL / (resolution * 1e-3))
                VOI_secondary = invert_z_axis(VOI_secondary)[:,:,:STOP_METAPHYSEAL_VOXELS]
                STOP_EPIPHYSEAL_VOXELS = int(STOP_EPIPHYSEAL / (resolution * 1e-3))
                VOI_epiphyseal = VOI_epiphyseal[:,:,:STOP_EPIPHYSEAL_VOXELS]
                if pred_zgp is not None:                    
                    VOI_mixed = invert_z_axis(im[:,:, :pred_zgp])[:,:,:STOP_METAPHYSEAL_VOXELS]
                else:
                    VOI_mixed = None
                
                #### Visualize all three only if they exist
                #### Make a list of non-None VOIs  
                # vis_images = [im for im in [VOI_secondary, VOI_mixed, VOI_epiphyseal] if im is not None]
                # titles_images = [t for t,im in zip([VOI_secondary, VOI_mixed, VOI_epiphyseal],["VOI Secondary", "VOI Mixed", "VOI Epiphyseal"]) if im is not None]
                # if not vis_images:
                #     print(f"No valid VOIs found for {bone_filename}. Skipping visualization.")
                #     continue
                # visualize_multiple_images(vis_images,
                #                        titles=titles_images
                #                        )
                
                sitk.WriteImage(VOI_secondary, os.path.join(output_bone_folder, bone_filename + "_VOI_secondary.nii.gz"))
                sitk.WriteImage(VOI_epiphyseal, os.path.join(output_bone_folder, bone_filename + "_VOI_epiphyseal.nii.gz"))
                if VOI_mixed is not None:
                    sitk.WriteImage(VOI_mixed, os.path.join(output_bone_folder, bone_filename + "_VOI_mixed.nii.gz"))
            except Exception as e:
                print(f"Error processing {bone_filename}: {e}")
                traceback.print_exc()
                continue