import numpy as np
import torch
import torch.nn.functional as F
import pose_estimation
import torch.nn as nn

# class ResBlock(nn.Module):
#     """
#     A single residual layer with three Conv-BN-ReLU stacks and a skip connection.
#     """
#     def __init__(self, channels=256, kernel_size=3, stride=1, downsample=False):
#         super(ResBlock, self).__init__()
  
#         new_stride = 2 if downsample else 1

#         self.conv1 = nn.Conv1d(channels, channels,
#                                kernel_size=kernel_size,
#                                stride=new_stride,
#                                padding=kernel_size // 2)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv1d(channels, channels,
#                                kernel_size=kernel_size,
#                                stride=1,
#                                padding=kernel_size // 2)
#         self.bn2 = nn.BatchNorm1d(channels)

#         self.conv3 = nn.Conv1d(channels, channels,
#                                kernel_size=kernel_size,
#                                stride=1,
#                                padding=kernel_size // 2)
#         self.bn3 = nn.BatchNorm1d(channels)
        
#         if downsample:
#           self.skip = nn.Sequential(
#             nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0),
#             nn.BatchNorm1d(channels)
#           )
#         else:
#           self.skip = nn.Identity()

#     def forward(self, x):
#         identity = self.skip(x)

#         # First conv + BN + ReLU
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         # # Second conv + BN + ReLU
#         # out = self.conv2(out)
#         # out = self.bn2(out)
#         # out = self.relu(out)

#         # # Third conv + BN + ReLU
#         # out = self.conv3(out)
#         # out = self.bn3(out)
#         # out = self.relu(out)

#         # Add skip connection
#         out = out + identity

#         out = self.relu(out)
#         return out

# class SquatResNet(nn.Module):
#     """
#     Implements the network from your figure (left):
#       1) conv(3, 2, 256) + BN + ReLU
#       2) 4 x ResBlock (the structure on the right)
#       3) AdaptiveAvgPool1d(1)
#       4) conv(1, 2, 256) + BN
#       5) Final output dimension is (batch_size, 256, 1) in the time axis
#          (assuming the stride=2 doesn't collapse it completely).

#     If you need a classification output (e.g. good vs. bad squat),
#     you can add a final linear layer or conv layer to produce 2 logits.
#     """
#     def __init__(self,
#                  input_channels=528,   # e.g., 3 if each "time step" has 3 features
#                  kernel_size=3,
#                  stride=2,
#                  base_channels=64,
#                  num_res_layers=2):
#         super(SquatResNet, self).__init__()

#         # 1) Initial Conv(3, 2, 256):
#         #    kernel_size=3, stride=2, out_channels=256
#         #    We'll pad = kernel_size//2 to maintain shape in convolution
#         self.conv1 = nn.Conv1d(input_channels, base_channels,
#                                kernel_size=kernel_size,
#                               #  stride=stride,
#                                stride=1,
#                               #  padding=kernel_size // 2
#                                padding=1)
#         self.bn1 = nn.BatchNorm1d(base_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.3)

#         # 2) 4 Residual Layers
#         layers = []
#         for i in range(num_res_layers):
#             down = (i % 2 == 0)
#             # down = True
#             layers.append(ResBlock(channels=base_channels, kernel_size=kernel_size, stride=1, downsample=down))
#             # if (i % 2 == 0):
#             #   layers.append(nn.Dropout(p=0.3))
        
#         layers.append(nn.Dropout(p=0.3))
#         self.res_layers = nn.Sequential(*layers)

#         # 3) Adaptive Average Pool -> output size = 1 in the time dimension
#         self.avgpool = nn.AdaptiveAvgPool1d(1)

#         # Optional: final classifier
#         # For a 2-class problem, you can add a linear layer or another conv:
#         self.fc = nn.Linear(base_channels, 2)
#         # or a final conv1d with out_channels=2, kernel_size=1, stride=1, etc.

#     def forward(self, x):
#         """
#         x shape: (batch_size, input_channels, time_length)
#         """
#         # 1) Initial conv + BN + ReLU
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         # 2) Residual layers
#         x = self.res_layers(x)

#         # 3) Adaptive average pooling -> (batch_size, base_channels, 1)
#         x = self.avgpool(x)

#         # You might want a final classifier. For example:
#         x = F.relu(x)
#         x = x.squeeze(-1)               # (batch_size, base_channels)
#         # x = self.dropout(x)
#         x = self.fc(x)            # (batch_size, 2)

#         return x
# class ResBlock(nn.Module):
#     """
#     A single residual layer with three Conv-BN-ReLU stacks and a skip connection.
#     """
#     def __init__(self, channels=256, kernel_size=3, stride=1, downsample=False):
#         super(ResBlock, self).__init__()

#         new_stride = 2 if downsample else 1

#         self.conv1 = nn.Conv1d(channels, channels,
#                                kernel_size=kernel_size,
#                                stride=new_stride,
#                                padding=kernel_size // 2)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv1d(channels, channels,
#                                kernel_size=kernel_size,
#                                stride=1,
#                                padding=kernel_size // 2)
#         self.bn2 = nn.BatchNorm1d(channels)

#         self.conv3 = nn.Conv1d(channels, channels,
#                                kernel_size=kernel_size,
#                                stride=1,
#                                padding=kernel_size // 2)
#         self.bn3 = nn.BatchNorm1d(channels)

#         if downsample:
#           self.skip = nn.Sequential(
#             nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0),
#             nn.BatchNorm1d(channels)
#           )
#         else:
#           self.skip = nn.Identity()

#     def forward(self, x):
#         identity = self.skip(x)

#         # First conv + BN + ReLU
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         # Second conv + BN + ReLU
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         # Third conv + BN + ReLU
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.relu(out)

#         # Add skip connection
#         out = out + identity

#         out = self.relu(out)
#         return out
  
# class SquatResNet(nn.Module):
#     """
#     Implements the network from your figure (left):
#       1) conv(3, 2, 256) + BN + ReLU
#       2) 4 x ResBlock (the structure on the right)
#       3) AdaptiveAvgPool1d(1)
#       4) conv(1, 2, 256) + BN
#       5) Final output dimension is (batch_size, 256, 1) in the time axis
#          (assuming the stride=2 doesn't collapse it completely).

#     If you need a classification output (e.g. good vs. bad squat),
#     you can add a final linear layer or conv layer to produce 2 logits.
#     """
#     def __init__(self,
#                  input_channels=528,   # e.g., 3 if each "time step" has 3 features
#                  kernel_size=3,
#                  stride=2,
#                  base_channels=128,
#                  num_res_layers=4):
#         super(SquatResNet, self).__init__()

#         # 1) Initial Conv(3, 2, 256):
#         #    kernel_size=3, stride=2, out_channels=256
#         #    We'll pad = kernel_size//2 to maintain shape in convolution
#         self.conv1 = nn.Conv1d(input_channels, base_channels,
#                                kernel_size=kernel_size,
#                               #  stride=stride,
#                                stride=1,
#                               #  padding=kernel_size // 2
#                                padding=1)
#         self.bn1 = nn.BatchNorm1d(base_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.4)

#         # 2) 4 Residual Layers
#         layers = []
#         for i in range(num_res_layers):
#             down = (i % 2 == 0)
#             # down = True
#             layers.append(ResBlock(channels=base_channels, kernel_size=kernel_size, stride=1, downsample=down))
#             # if (i % 2 == 0):
#             #   layers.append(nn.Dropout(p=0.3))

#         # layers.append(nn.Dropout(p=0.3))
#         self.res_layers = nn.Sequential(*layers)

#         # 3) Adaptive Average Pool -> output size = 1 in the time dimension
#         self.avgpool = nn.AdaptiveAvgPool1d(1)

#         # Optional: final classifier
#         # For a 2-class problem, you can add a linear layer or another conv:
#         self.fc = nn.Linear(base_channels, 2)
#         # or a final conv1d with out_channels=2, kernel_size=1, stride=1, etc.

#     def forward(self, x):
#         """
#         x shape: (batch_size, input_channels, time_length)
#         """
#         # 1) Initial conv + BN + ReLU
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.dropout(x)

#         # 2) Residual layers
#         x = self.res_layers(x)
#         x = self.dropout(x)

#         # 3) Adaptive average pooling -> (batch_size, base_channels, 1)
#         x = self.avgpool(x)

#         # You might want a final classifier. For example:
#         x = F.relu(x)
#         x = x.squeeze(-1)               # (batch_size, base_channels)
#         x = self.fc(x)            # (batch_size, 2)

#         return x

class ResBlock(nn.Module):
    """
    A single residual layer with three Conv-BN-ReLU stacks and a skip connection.
    """
    def __init__(self, channels=256, kernel_size=3, stride=1, downsample=False):
        super(ResBlock, self).__init__()

        new_stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(channels, channels,
                               kernel_size=kernel_size,
                               stride=new_stride,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(channels, channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)

        self.conv3 = nn.Conv1d(channels, channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(channels)

        if downsample:
          self.skip = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm1d(channels)
          )
        else:
          self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv + BN + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Third conv + BN + ReLU
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # Add skip connection
        out = out + identity

        out = self.relu(out)
        return out

class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        # First dilated conv
        self.conv1 = nn.Conv1d(channels, channels,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        # Second dilated conv (same params)
        self.conv2 = nn.Conv1d(channels, channels,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        # Final activation
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = out + identity
        return self.relu2(out)

class SquatResNet(nn.Module):
    """
    Implements the network from your figure (left):
      1) conv(3, 2, 256) + BN + ReLU
      2) 4 x ResBlock (the structure on the right)
      3) AdaptiveAvgPool1d(1)
      4) conv(1, 2, 256) + BN
      5) Final output dimension is (batch_size, 256, 1) in the time axis
         (assuming the stride=2 doesn't collapse it completely).

    If you need a classification output (e.g. good vs. bad squat),
    you can add a final linear layer or conv layer to produce 2 logits.
    """
    def __init__(self,
                 input_channels=528,   # e.g., 3 if each "time step" has 3 features
                 kernel_size=3,
                 stride=2,
                 base_channels=128,
                 num_res_layers=4):
        super(SquatResNet, self).__init__()

        # 1) Initial Conv(3, 2, 256):
        #    kernel_size=3, stride=2, out_channels=256
        #    We'll pad = kernel_size//2 to maintain shape in convolution
        self.conv1 = nn.Conv1d(input_channels, base_channels,
                               kernel_size=kernel_size,
                              #  stride=stride,
                               stride=1,
                              #  padding=kernel_size // 2
                               padding=1)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)

        # 2) 4 Residual Layers
        layers = []
        for i in range(num_res_layers//2):
            down = (i % 2 == 0)
            # down = True
            layers.append(ResBlock(channels=base_channels, kernel_size=kernel_size, stride=1, downsample=down))
            # if (i % 2 == 0):
            #   layers.append(nn.Dropout(p=0.3))
          
        for i in range(2, 4):
            layers.append(DilatedResBlock(channels=base_channels, kernel_size=kernel_size, dilation=i))

        # layers.append(nn.Dropout(p=0.3))
        self.res_layers = nn.Sequential(*layers)

        # 3) Adaptive Average Pool -> output size = 1 in the time dimension
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Optional: final classifier
        # For a 2-class problem, you can add a linear layer or another conv:
        self.fc = nn.Linear(base_channels, 2)
        # or a final conv1d with out_channels=2, kernel_size=1, stride=1, etc.

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, time_length)
        """
        # 1) Initial conv + BN + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        # 2) Residual layers
        x = self.res_layers(x)
        x = self.dropout(x)

        # 3) Adaptive average pooling -> (batch_size, base_channels, 1)
        x = self.avgpool(x)

        # You might want a final classifier. For example:
        x = F.relu(x)
        x = x.squeeze(-1)               # (batch_size, base_channels)
        x = self.fc(x)            # (batch_size, 2)

        return x


def classify(mat):
    if isinstance(mat, np.ndarray) and mat.dtype == np.object_:
      mat = np.array(mat.tolist(), dtype=np.float32)  # convert from object array to regular float array
    else:
      mat = np.array(mat, dtype=np.float32)
    t = np.linspace(0, 1, mat.shape[1])
    mat = np.vstack([mat, t])
    mat = torch.tensor(mat, dtype=torch.float32)
    mat = (mat - mat.mean(dim=1, keepdim=True)) / (mat.std(dim=1, keepdim=True) + 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SquatResNet(input_channels=529).to(device)
    model.load_state_dict(torch.load("backend/models/model_TCNN.pth", map_location=device))  # make sure model is saved first
    model.eval()

    # Load the new pose matrix (.npy file) for inference
    tensor = mat.unsqueeze(0).to(device)  # shape: (1, 528, T)

    # Run the model to get prediction
    with torch.no_grad():
        logits = model(tensor)  # shape: (1, 2)
        probs = F.softmax(logits, dim=1)  # convert logits to probabilities
        print(probs)
        pred = torch.argmax(probs, dim=1).item()  # get predicted class (0 or 1)
        conf = probs[0][pred].item()  # get confidence of the predicted class

    # Map prediction to label
    label_map = {0: "Bad Squat", 1: "Good Squat"}
    print(f"✅ Your squat is {conf * 100:.2f}% {label_map[pred].lower()}.")
  

if __name__ == "__main__":
    mat, results_dict = pose_estimation.get_video_data("/Users/mukundmaini/Downloads/IMG_2944.MOV", save_vid=False)
    classify(mat)