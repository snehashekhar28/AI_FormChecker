import mediapipe as mp
import cv2
import numpy as np
print(mp.__version__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
video_path = "./FormChecker/test_videos/squat_7.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#represents the 2d matrix where each pose/landmark of a frame is a column and each frame itself
#is a row
frame_pose_vectors = []
file = open()
# Function to normalize pose using hip width
def normalize_pose(landmarks):
    """ Normalizes the pose by scaling joints to a unit length based on the hip width. """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    # Compute reference length (hip width)
    ref_length = np.linalg.norm(np.array([left_hip.x - right_hip.x, 
                                          left_hip.y - right_hip.y, 
                                          left_hip.z - right_hip.z]))
    print(ref_length)

    if ref_length == 0:  # Avoid division by zero
        return None

    # Normalize all landmarks by dividing by the reference length
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append([
            (landmark.x - left_hip.x) / ref_length,  # Normalize and center around left hip
            (landmark.y - left_hip.y) / ref_length,
            (landmark.z - left_hip.z) / ref_length
        ])
    
    return np.array(normalized_landmarks)


while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Process frame with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        #I believe number of landmarks for us is 32, research paper says they only have 19
        num_landmarks = len(landmarks)
        # Normalize the pose
        normalized_landmarks = normalize_pose(landmarks)
        if normalized_landmarks is None:
            continue
        #Calculates the euclidean distance 
        distance_matrix = np.zeros((num_landmarks, num_landmarks))
        distance_vector = []
        for i in range(num_landmarks):
            # j starts at i + 1 so that we're only looking at the upper right triangle of the matrix
            for j in range(i + 1, num_landmarks):
                distance = np.linalg.norm(normalized_landmarks[i] - normalized_landmarks[j])
                distance_vector.append(distance)  # Append to vector
         #adding the whole frame to the 2d table           
        frame_pose_vectors.append(distance_vector)
    # Display the frame
    cv2.imshow("MediaPipe Pose", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
    frame_count += 1
# Release resources
frame_pose_matrix = np.array(frame_pose_vectors).T
print("\nFrame vs. Distance Matrix (Time-Series Representation of Pose):")
print(frame_pose_matrix)
print(f"Matrix Shape: {frame_pose_matrix.shape}")  # (Total frames × Distance features
cap.release()
cv2.destroyAllWindows()