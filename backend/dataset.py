from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pose_estimation

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
# Traverse the directory structure
for root, dirs, files in os.walk(video_path):
    if error:
        break
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions if needed
            print("VID:", cnt)
            cnt += 1
            video_path = os.path.join(root, file)
            print("file:", video_path)
            try:
                mat = pose_estimation.get_video_data(video_path=video_path, save_vid=False)
                if len(mat) != 0:
                    mats.append(mat)
                    if "good" in video_path and "bad" not in video_path:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    bad_vid_paths.append(video_path)
                    print("**********************BAD VID************************")
            except:
                print("*************ERROR, SAVING FILES***********************")
                error = True
                np.save("/Users/mukundmaini/Downloads/Video_Matrices/mats.npy", np.array(mats, dtype=object))
                np.save("/Users/mukundmaini/Downloads/Video_Matrices/labels.npy", np.array(labels))
                np.save("/Users/mukundmaini/Downloads/Video_Matrices/bad_vids.npy", np.array(bad_vid_paths, dtype=str))
                break

print("num bad vids:", len(bad_vid_paths))
if not error:
    np.save("/Users/mukundmaini/Downloads/Video_Matrices/mats.npy", np.array(mats, dtype=object))
    np.save("/Users/mukundmaini/Downloads/Video_Matrices/labels.npy", np.array(labels))
    np.save("/Users/mukundmaini/Downloads/Video_Matrices/bad_vids.npy", np.array(bad_vid_paths, dtype=str))
# dataset = SquatDataset(mats, labels)
# just store mats and labels and load in collab notebook because can only load dataset in same 
# environment it was created in