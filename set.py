import synapseclient 
import synapseutils 
from dotenv import load_dotenv
import os, shutil, glob
import re

load_dotenv()

# Create necessary directories
data_root = "./data/synapse"
raw_dir = os.path.join(data_root, "Abdomen/RawData")

# Step 1: Download from Synapse if necessary
# Check if directory exists and is empty (or doesn't exist)
should_download = False
if should_download:
    print("Downloading data from Synapse...")
    # Create raw_dir only when we need to download
    os.makedirs(raw_dir, exist_ok=True)
    
    syn = synapseclient.Synapse() 
    syn.login(authToken=os.getenv('SYNAPSE_KEY')) 
    files = synapseutils.syncFromSynapse(syn, 'syn3193805') 
    
    for f in files:
        local_path = f['path']
        filename = os.path.basename(local_path)
        shutil.copy(local_path, os.path.join(raw_dir, filename))
    print(f"Downloaded {len(files)} files from Synapse")
else:
    print(f"Using existing files in {raw_dir}")

# Now create the other directories after handling the download
train_img_dir = os.path.join(raw_dir, "TrainSet", "img")
train_label_dir = os.path.join(raw_dir, "TrainSet", "label")
test_img_dir = os.path.join(raw_dir, "TestSet", "img")
test_label_dir = os.path.join(raw_dir, "TestSet", "label")

for d in [train_img_dir, train_label_dir, test_img_dir, test_label_dir]:
    os.makedirs(d, exist_ok=True)

# Step 2: Clean up unpaired files
nii_files = sorted(glob.glob(os.path.join(raw_dir, "*_avg.nii.gz")))
removed = 0
for img_path in nii_files:
    seg_path = img_path.replace("_avg.nii.gz", "_avg_seg.nii.gz")
    if not os.path.exists(seg_path):
        print(f"❌ Removing {os.path.basename(img_path)} (missing seg)")
        os.remove(img_path)
        removed += 1
print(f"Cleanup done. Removed {removed} unpaired images.")

# Print list of files for debugging
print(f"Files in raw_dir: {os.listdir(raw_dir)}")
print(f"NII files found: {[os.path.basename(f) for f in nii_files]}")

# Step 3: Read test cases
with open("test_vol.txt") as f:
    test_cases = [line.strip() for line in f]
print(f"Found {len(test_cases)} test cases: {test_cases}")

# Step 4: Create a mapping between synapse files and case IDs
img_files = sorted(glob.glob(os.path.join(raw_dir, "*_avg.nii.gz")))
file_to_case = {}

# Using numerical order to assign case IDs in order
if len(img_files) > 0:
    # Extract all numeric IDs from filenames
    all_nums = []
    for img_file in img_files:
        filename = os.path.basename(img_file)
        match = re.search(r'(\d+)', filename)
        if match:
            all_nums.append(int(match.group(1)))
    
    # Sort the numbers
    all_nums = sorted(all_nums)
    
    # Create a mapping from numbers to case IDs
    num_to_case = {}
    for i, num in enumerate(sorted(set(all_nums))):  # Use set to handle duplicates
        case_id = f"case{i+1:04d}"  # Start with case0001
        num_to_case[num] = case_id
    
    # Now map files to case IDs
    for img_file in img_files:
        filename = os.path.basename(img_file)
        match = re.search(r'(\d+)', filename)
        if match:
            num = int(match.group(1))
            file_to_case[filename] = num_to_case[num]
    
    print("\nFile to case mapping:")
    for f, c in file_to_case.items():
        print(f"{f} -> {c}")

# Step 5: Manually map test_cases to files
# Based on the sequential ordering we used above
test_case_indices = [int(case[4:]) for case in test_cases]  # Extract numbers from case0008 -> 8
actual_test_cases = []

for idx, (file, case) in enumerate(file_to_case.items(), 1):
    if idx in test_case_indices:
        actual_test_cases.append(file)

print(f"\nIdentified {len(actual_test_cases)} test files: {actual_test_cases}")

# Step 6: Organize files into train/test directories
for fname in os.listdir(raw_dir):
    if not fname.endswith(".nii.gz"):
        continue
        
    # Get base filename without extension
    base = fname.replace(".nii.gz", "")
    is_seg = "_seg" in base
    
    # Get corresponding image filename for seg files
    if is_seg:
        img_fname = fname.replace("_seg.nii.gz", ".nii.gz")
    else:
        img_fname = fname
        
    # Skip if no mapping exists
    if img_fname not in file_to_case:
        print(f"⚠️ No mapping found for {fname}, skipping")
        continue
    
    # Decide destination based on image filename (not segmentation)
    if img_fname in actual_test_cases:
        if is_seg:
            # For labels in test set
            case_num = file_to_case[img_fname][4:]  # Extract "0001" from "case0001"
            dest_filename = f"label{case_num}.nii.gz"  # label0001.nii.gz
            dest_path = os.path.join(test_label_dir, dest_filename)
            print(f"Copying {fname} to test labels as {dest_filename}")
            shutil.copy(os.path.join(raw_dir, fname), dest_path)
        else:
            # For images in test set
            case_num = file_to_case[img_fname][4:]  # Extract "0001" from "case0001"
            dest_filename = f"img{case_num}.nii.gz"  # img0001.nii.gz
            dest_path = os.path.join(test_img_dir, dest_filename)
            print(f"Copying {fname} to test images as {dest_filename}")
            shutil.copy(os.path.join(raw_dir, fname), dest_path)
    else:
        if is_seg:
            # For labels in train set
            case_num = file_to_case[img_fname][4:]  # Extract "0001" from "case0001"
            dest_filename = f"label{case_num}.nii.gz"  # label0001.nii.gz
            dest_path = os.path.join(train_label_dir, dest_filename)
            print(f"Copying {fname} to train labels as {dest_filename}")
            shutil.copy(os.path.join(raw_dir, fname), dest_path)
        else:
            # For images in train set
            case_num = file_to_case[img_fname][4:]  # Extract "0001" from "case0001"
            dest_filename = f"img{case_num}.nii.gz"  # img0001.nii.gz
            dest_path = os.path.join(train_img_dir, dest_filename)
            print(f"Copying {fname} to train images as {dest_filename}")
            shutil.copy(os.path.join(raw_dir, fname), dest_path)

# Paths to preprocessed data
train_npz_dir = os.path.join(data_root, "train_npz_new")
test_vol_h5_dir = os.path.join(data_root, "test_vol_h5_new")

# Output list paths
lists_dir = "./lists/lists_Synapse"
os.makedirs(lists_dir, exist_ok=True)
train_list_path = os.path.join(lists_dir, "train.txt")
test_list_path = os.path.join(lists_dir, "test.txt")

# Regenerate train.txt
with open(train_list_path, "w") as f:
    for file in sorted(os.listdir(train_npz_dir)):
        if file.endswith(".npz"):
            f.write(file.replace(".npz", "") + "\n")
print(f"Regenerated train list at {train_list_path}")

# Regenerate test.txt (for test volumes, if needed)
with open(test_list_path, "w") as f:
    for file in sorted(os.listdir(test_vol_h5_dir)):
        if file.endswith(".h5"):
            f.write(file.replace(".h5", "") + "\n")
print(f"Regenerated test list at {test_list_path}")

print("\nFile organization and list regeneration complete.")
print(f"Train images: {len(os.listdir(train_img_dir))}")
print(f"Train labels: {len(os.listdir(train_label_dir))}")
print(f"Test images: {len(os.listdir(test_img_dir))}")
print(f"Test labels: {len(os.listdir(test_label_dir))}")
print(f"Train slices: {len(os.listdir(train_npz_dir))}")
print(f"Test volumes: {len(os.listdir(test_vol_h5_dir))}")

print("\nNext step: Run the preprocessing script if not done yet:")
print("python ./utils/preprocess_synapse_data.py")
print("Then you can run training and testing scripts smoothly.")