import mediapipe as mp
import cv2
import csv
import os
from tqdm import tqdm

# =======================
# Configurations
# =======================
image_folder = "images"   # Change this to your dataset path
output_csv = "face_keypoints.csv"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Landmark indices: 
LANDMARK_INDICES = [1, 33, 263, 61, 291]

# =======================
# Collect image files
# =======================
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# =======================
# Process images & save CSV
# =======================
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    header = ['image_name']
    for i in range(len(LANDMARK_INDICES)):
        header.append(f'point_{i+1}_x')
        header.append(f'point_{i+1}_y')
    csv_writer.writerow(header)

    # Process each image
    for image_name in tqdm(image_files, desc="Extracting Keypoints"):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        results = face_mesh.process(image_rgb)

        # Start row with image name
        row = [image_name]

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            total_landmarks = len(face_landmarks.landmark)

            for idx in LANDMARK_INDICES:
                if idx < total_landmarks:
                    lm = face_landmarks.landmark[idx]
                    # Normalized coords (0-1), can multiply by width/height if needed
                    row.append(lm.x)
                    row.append(lm.y)
                else:
                    row.extend([-1.0, -1.0])  # Missing point
        else:
            # No face detected → fill with -1
            row.extend([-1.0] * (len(LANDMARK_INDICES) * 2))

        # Write row
        csv_writer.writerow(row)

print("✅ Keypoint extraction complete! Saved to", output_csv)

# Close MediaPipe
face_mesh.close()
