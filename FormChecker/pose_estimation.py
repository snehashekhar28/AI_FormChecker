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
        
        #Calculates the euclidean distance 
        distance_matrix = np.zeros((num_landmarks, num_landmarks))
        distance_vector = []
        for i in range(num_landmarks):
            for j in range(i + 1, num_landmarks):
                xi, yi, zi, = landmarks[i].x, landmarks[i].y, landmarks[i].z
                xj, yj, zj = landmarks[j].x, landmarks[j].y, landmarks[j].z
                distance = np.sqrt((xi - xj) ** 2 + (yi -yj) ** 2 + (zi - zj) ** 2)
                #populating the distance matrix
                distance_matrix[i, j] = distance
                #flattening out the distance matrix into a vector
                if (distance != 0):
                    distance_vector.append(distance)
         #adding the whole frame to the 2d table           
        frame_pose_vectors.append(distance_vector)
    # Display the frame
    cv2.imshow("MediaPipe Pose", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
    frame_count += 1
# Release resources
frame_pose_matrix = np.array(frame_pose_vectors)
print("\nFrame vs. Distance Matrix (Time-Series Representation of Pose):")
print(frame_pose_matrix)
print(f"Matrix Shape: {frame_pose_matrix.shape}")  # (Total frames × Distance features
cap.release()
cv2.destroyAllWindows()