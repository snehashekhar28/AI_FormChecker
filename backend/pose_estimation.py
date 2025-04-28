import mediapipe as mp
import cv2
import numpy as np
import os
import json
import dotenv
from openai import OpenAI
import json

dotenv.load_dotenv()

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

def calculate_torso_angle(landmarks, mp_pose):
    """Returns torso lean angle with respect to vertical."""
    shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
    hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
    vertical = np.array([0, -1])
    torso_vector = shoulder - hip
    angle_rad = np.arccos(np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector)))
    return np.degrees(angle_rad)

def hip_below_knee(landmarks, mp_pose):
    """Checks if hip is below knee (squat depth)."""
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    return left_hip_y > left_knee_y  # y increases downwards in image space

def ankle_dorsiflexion(landmarks, mp_pose, side="left"):
    """Angle between foot and shin to assess ankle mobility."""
    if side == "left":
        ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y])
        knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y])
        toe = np.array([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y])
    else:
        ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y])
        knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y])
        toe = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y])

    shin = knee - ankle
    foot = toe - ankle
    dot = np.dot(shin, foot)
    norms = np.linalg.norm(shin) * np.linalg.norm(foot)
    if norms == 0:
        return None
    angle_rad = np.arccos(dot / norms)
    return np.degrees(angle_rad)
# Function to normalize pose using hip width
def calculate_knee_angle(landmarks, mp_pose, side="left"):
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


def normalize_pose(landmarks, mp_pose):
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


def get_video_data(video_path, save_vid=False, save_path='./rep_only_video.mp4', display_vid=True):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #represents the 2d matrix where each pose/landmark of a frame is a column and each frame itself
    #is a row
    frame_pose_vectors = []
    prev_knee_angle = (0, 0)
    tracking_rep = False
    done_with_one_rep = False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_left_knee_angle = 30000
    min_right_knee_angle = 30000
    min_torso_angle = 180
    min_ankle_dorsi_left = 180
    min_ankle_dorsi_right = 180
    hip_below_knee_detected = False

    # Check for rotated video
    test_ret, test_frame = cap.read()
    h, w = test_frame.shape[:2]
    do_rotate = (w > h)

    # Create video writer to save rep-only video
    if save_vid:
        if do_rotate:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (h, w))
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break

        if do_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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
            torso_angle = calculate_torso_angle(landmarks, mp_pose)
            min_torso_angle = min(min_torso_angle, torso_angle)
            left_knee_angle = calculate_knee_angle(landmarks, mp_pose, side="left")
            right_knee_angle = calculate_knee_angle(landmarks, mp_pose, side="right")
            ankle_left = ankle_dorsiflexion(landmarks, mp_pose, "left")
            ankle_right = ankle_dorsiflexion(landmarks, mp_pose, "right")
            if ankle_left: min_ankle_dorsi_left = min(min_ankle_dorsi_left, ankle_left)
            if ankle_right: min_ankle_dorsi_right = min(min_ankle_dorsi_right, ankle_right)

            if hip_below_knee(landmarks, mp_pose):
                hip_below_knee_detected = True
            if (left_knee_angle < min_left_knee_angle):
                min_left_knee_angle = left_knee_angle
            if (right_knee_angle < min_right_knee_angle):
                min_right_knee_angle = right_knee_angle
            # Use the larger knee angle (whichever is more bent)
            knee_angle = left_knee_angle + right_knee_angle
            prev_greater = (prev_knee_angle[0] + prev_knee_angle[1] >= 280) or (max(prev_knee_angle[0], prev_knee_angle[1]) >= 140 and abs(prev_knee_angle[0] - prev_knee_angle[1]) >= 25)
            curr_less = (knee_angle < 280) or ((left_knee_angle < 145 or right_knee_angle < 145) and abs(left_knee_angle - right_knee_angle) >= 25)
            curr_greater = (left_knee_angle + right_knee_angle >= 280) or (max(left_knee_angle, right_knee_angle) >= 140 and abs(left_knee_angle - right_knee_angle) >= 25)

            # Detect rep start (transition from standing to squat)
            if prev_greater and curr_less:
                tracking_rep = True  # Start tracking

            # Detect rep end (transition back to standing)
            if tracking_rep and curr_greater and len(frame_pose_vectors) > 30:
                tracking_rep = False  # Stop tracking
                done_with_one_rep = True
            # prev_knee_angle = knee_angle  # Update previous angle
            prev_knee_angle = (left_knee_angle, right_knee_angle)
            if done_with_one_rep:
                break
            if tracking_rep and not done_with_one_rep:
                if save_vid:
                    out.write(frame)
                num_landmarks = len(landmarks)
                normalized_landmarks = normalize_pose(landmarks, mp_pose)
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
        if display_vid:
            cv2.imshow("MediaPipe Pose", frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        frame_count += 1

    # Release resources
    frame_pose_matrix = np.array(frame_pose_vectors).T
    if len(frame_pose_matrix) != 0:
        frame_pose_matrix = interpolate_frames(frame_pose_matrix) 
    print("\nFrame vs. Distance Matrix (Time-Series Representation of Pose):")
    print(frame_pose_matrix)
    print(f"Matrix Shape: {frame_pose_matrix.shape}")  # (Total frames × Distance features
    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("save?", save_vid)
    print("saved annotated to:", save_path)
    results_dict = {"min_left_knee_angle": min_left_knee_angle, 
                    "min_right_knee_angle": min_right_knee_angle,
                    "min_torso_angle": min_torso_angle,
                    "min_ankle_dorsiflexion_left": min_ankle_dorsi_left,
                    "min_ankle_dorsiflexion_right": min_ankle_dorsi_right,
                    "is_hip_below_knee": hip_below_knee_detected}

    return frame_pose_matrix, results_dict



def generate_natural_language_feedback(results_dict):

    prompt = f"""
    You're a squat coach. I’m giving you 3D pose feature data from a squat rep. Here's the vector:

    {json.dumps(results_dict, indent=2)}

    Analyze the form based on this, but be extremely brief — just 2-3 short high-level tips. MAX, 30 words. Do NOT explain why something is bad or what specific angles mean. Just mention what could be off, in natural language, as if you're texting a friend. No labeling of angles or joint names directly. Just make it sound like quick, casual advice.

    Do NOT summarize or list things in detail. Do NOT say “based on the data.” Just give 2-3 quick tips that someone could try fixing.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    video_path = "/Users/mukundmaini/Downloads/IMG_2943.MOV"
    frame_pose_matrix, results_dict = get_video_data(video_path=video_path, save_vid=False)
    print(generate_natural_language_feedback(results_dict))
