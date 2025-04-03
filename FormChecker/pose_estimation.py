import mediapipe as mp
import cv2
import numpy as np
print(mp.__version__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
video_path = "./FormChecker/test_videos/squat_11.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#represents the 2d matrix where each pose/landmark of a frame is a column and each frame itself
#is a row
frame_pose_vectors = []
prev_knee_angle = 180 
tracking_rep = False

# Function to normalize pose using hip width

def calculate_knee_angle(landmarks, side="left"):
    """Calculate the knee angle using hip, knee, and ankle landmarks."""
    if side == "left":
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    else:
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Convert to NumPy arrays
    hip = np.array([hip.x, hip.y, hip.z])
    knee = np.array([knee.x, knee.y, knee.z])
    ankle = np.array([ankle.x, ankle.y, ankle.z])

    v1 = hip - knee  # Thigh vector
    v2 = ankle - knee  # Shin vector

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None  # Invalid case

    angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def normalize_pose(landmarks):
    """ Normalizes the pose by scaling joints to a unit length based on the hip width. """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    # Compute reference length (hip width)
    ref_length = np.linalg.norm(np.array([left_hip.x - right_hip.x, 
                                          left_hip.y - right_hip.y, 
                                          left_hip.z - right_hip.z]))
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


def interpolate_frames(frame_pose_vectors, F_fixed=100):
    """
    Resamples the frame_pose_vectors to have exactly F_fixed frames using linear interpolation.
    
    Parameters:
        frame_pose_vectors (numpy.ndarray): Shape (528, F_original) where F_original is variable.
        F_fixed (int): The target number of frames.

    Returns:
        numpy.ndarray: Shape (528, F_fixed)
    """
    F_original = frame_pose_vectors.shape[1]
    
    # Generate target indices for interpolation
    target_indices = np.linspace(0, F_original - 1, F_fixed)
    
    # Interpolate each row (distance vector per landmark pair)
    interpolated_matrix = np.zeros((frame_pose_vectors.shape[0], F_fixed))
    for i in range(frame_pose_vectors.shape[0]):
        interpolated_matrix[i, :] = np.interp(target_indices, np.arange(F_original), frame_pose_vectors[i, :])

    return interpolated_matrix



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


        # Calculate knee angle
        left_knee_angle = calculate_knee_angle(landmarks, side="left")
        right_knee_angle = calculate_knee_angle(landmarks, side="right")
        print("left_knee_angle: " + str(left_knee_angle))
        print("right_knee_angle: " + str(right_knee_angle))
        # Use the smaller knee angle (whichever is more bent)
        knee_angle = min(left_knee_angle, right_knee_angle)

        # Detect rep start (transition from standing to squat)
        if prev_knee_angle >= 150 and knee_angle < 150:
            tracking_rep = True  # Start tracking

        # Detect rep end (transition back to standing)
        if tracking_rep and knee_angle >= 150:
            tracking_rep = False  # Stop tracking

        prev_knee_angle = knee_angle  # Update previous angle
        
        if tracking_rep:
            num_landmarks = len(landmarks)
            normalized_landmarks = normalize_pose(landmarks)
            if normalized_landmarks is None:
                continue

            distance_vector = []
            for i in range(num_landmarks):
                for j in range(i + 1, num_landmarks):  # Only upper triangle
                    distance = np.linalg.norm(normalized_landmarks[i] - normalized_landmarks[j])
                    distance_vector.append(distance)

            frame_pose_vectors.append(distance_vector)  # Add to dataset
         #adding the whole frame to the 2d table           
    # Display the frame
    cv2.imshow("MediaPipe Pose", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
    frame_count += 1

# Release resources
frame_pose_matrix = np.array(frame_pose_vectors).T
frame_pose_matrix = interpolate_frames(frame_pose_matrix)
print("\nFrame vs. Distance Matrix (Time-Series Representation of Pose):")
print(frame_pose_matrix)
print(f"Matrix Shape: {frame_pose_matrix.shape}")  # (Total frames × Distance features
cap.release()
cv2.destroyAllWindows()