from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pose_estimation
import glob
import random

class SquatDataset(Dataset):
    def __init__(self, mats, labels, max_frames=100):
        self.tensors = []
        for mat in mats:
          mat = torch.tensor(mat, dtype=torch.float32)
          self.tensors.append(mat)
        
        self.label_numbers = torch.tensor(labels, dtype=torch.long)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        mat = self.tensors[idx]
        label_nums = self.label_numbers[idx]

        return mat, label_nums


video_path = "/Users/mukundmaini/Downloads/Video_Dataset"
mats = []
labels = []
bad_vid_paths = []
error = False
cnt = 1
good_dir = os.path.join(video_path, "good", "1115_video")
bad_categories = [
    "bad_back_round",
    "bad_back_warp",
    "bad_head",
    "bad_innner_thigh",
    "bad_shallow",
    "bad_toe",
]

# get all good files
good_paths_all = glob.glob(os.path.join(good_dir, "*.mp4"))
good_paths = []
for path in good_paths_all:
    if "good" in path and "bad" not in path:
        good_paths.append(path)

bad_paths_by_cat = {
    cat: glob.glob(os.path.join(video_path, cat, "1115_video", "*.mp4"))
    for cat in bad_categories
}

total_good = len(good_paths)
n_bad_cats = len(bad_categories)
print(total_good)
print(n_bad_cats)

# base count + distribute remainder
per_cat = total_good // n_bad_cats           # 272 // 6 = 45
remainder = total_good % n_bad_cats          # 272 % 6 = 2

# build a dict of how many to sample from each
sample_counts = {}
for i, cat in enumerate(bad_categories):
    extra = 1 if i < remainder else 0      # give one extra to the first `remainder` cats
    sample_counts[cat] = per_cat + extra

print(sample_counts)

random.seed(1097)  # for reproducibility
bad_sampled = []
for cat, paths in bad_paths_by_cat.items():
    k = sample_counts[cat]
    if k > len(paths):
        raise ValueError(f"Not enough videos in {cat} to sample {k}")
    bad_sampled += random.sample(paths, k)

print(len(bad_sampled))

all_paths = good_paths + bad_sampled
labels = [1]*len(good_paths) + [0]*len(bad_sampled)  # 1=good, 0=bad

# shuffle in unison
combined = list(zip(all_paths, labels))
random.shuffle(combined)
all_paths, labels = zip(*combined)
labels = list(labels)
print(len(all_paths))
print(len(labels))

for i, file in enumerate(all_paths):
    print("VID:", cnt)
    cnt += 1
    print("file:", file)
    try:
        mat = pose_estimation.get_video_data(video_path=file, save_vid=False)
        if len(mat) != 0:
            mats.append(mat)
        else:
            bad_vid_paths.append(video_path)
            del labels[i]
            print("**********************BAD VID************************")
    except:
        print("*************ERROR, SAVING FILES***********************")
        error = True
        np.save("/Users/mukundmaini/Downloads/Video_Matrices/mats_smal3.npy", np.array(mats, dtype=object))
        np.save("/Users/mukundmaini/Downloads/Video_Matrices/labels_small3.npy", np.array(labels))
        break

print(bad_vid_paths)
print("num bad vids:", len(bad_vid_paths))
print("num mats:", len(mats))
print("num labels:", len(labels))
if not error:
    np.save("/Users/mukundmaini/Downloads/Video_Matrices/mats_small3.npy", np.array(mats, dtype=object))
    np.save("/Users/mukundmaini/Downloads/Video_Matrices/labels_small3.npy", np.array(labels))

# Traverse the directory structure
# for root, dirs, files in os.walk(video_path):
#     if error:
#         break
#     for file in files:
#         if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions if needed
#             print("VID:", cnt)
#             cnt += 1
#             video_path = os.path.join(root, file)
#             print("file:", video_path)
#             try:
#                 mat = pose_estimation.get_video_data(video_path=video_path, save_vid=False)
#                 if len(mat) != 0:
#                     mats.append(mat)
#                     if "good" in video_path and "bad" not in video_path:
#                         labels.append(1)
#                     else:
#                         labels.append(0)
#                 else:
#                     bad_vid_paths.append(video_path)
#                     print("**********************BAD VID************************")
#             except:
#                 print("*************ERROR, SAVING FILES***********************")
#                 error = True
#                 np.save("/Users/mukundmaini/Downloads/Video_Matrices/mats.npy", np.array(mats, dtype=object))
#                 np.save("/Users/mukundmaini/Downloads/Video_Matrices/labels.npy", np.array(labels))
#                 np.save("/Users/mukundmaini/Downloads/Video_Matrices/bad_vids.npy", np.array(bad_vid_paths, dtype=str))
#                 break

# print("num bad vids:", len(bad_vid_paths))
# if not error:
#     np.save("/Users/mukundmaini/Downloads/Video_Matrices/mats.npy", np.array(mats, dtype=object))
#     np.save("/Users/mukundmaini/Downloads/Video_Matrices/labels.npy", np.array(labels))
#     np.save("/Users/mukundmaini/Downloads/Video_Matrices/bad_vids.npy", np.array(bad_vid_paths, dtype=str))
# dataset = SquatDataset(mats, labels)
# just store mats and labels and load in collab notebook because can only load dataset in same 
# environment it was created in