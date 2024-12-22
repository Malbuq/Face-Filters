import cv2
import numpy as np
import mediapipe as mp

def print_png(frame, png_path, png_coords):
    mustache_image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

    # Split PNG into RGB and alpha channels
    mustache_rgb = mustache_image[:, :, :3]
    mustache_alpha = mustache_image[:, :, 3] / 255.0

    h, w, _ = frame.shape

    png_x, png_y, roll_angle, yaw_angle, pitch_angle = png_coords

    # Resize mustache
    mustache_width = w // 8  # Adjust width as needed
    mustache_height = int(mustache_width * (mustache_rgb.shape[0] / mustache_rgb.shape[1]))
    resized_mustache = cv2.resize(mustache_rgb, (mustache_width, mustache_height))
    resized_alpha = cv2.resize(mustache_alpha, (mustache_width, mustache_height))

    # Scale coordinates to the frame size
    center_x = int(png_x * w)
    center_y = int(png_y * h)

    # Calculate top-left corner for overlay
    top_left_x = center_x - mustache_width // 2
    top_left_y = center_y - mustache_height // 2 + 25

    # Apply roll rotation
    rotation_matrix = cv2.getRotationMatrix2D((mustache_width // 2, mustache_height // 2), roll_angle, 1)
    rotated_mustache = cv2.warpAffine(resized_mustache, rotation_matrix, (mustache_width, mustache_height))
    rotated_alpha = cv2.warpAffine(resized_alpha, rotation_matrix, (mustache_width, mustache_height))

    # Simulate yaw and pitch using perspective warp
    yaw_factor = np.tan(np.radians(yaw_angle)) * mustache_width // 2
    pitch_factor = np.tan(np.radians(pitch_angle)) * mustache_height // 2
    src_points = np.float32([
        [0, 0],
        [mustache_width, 0],
        [mustache_width, mustache_height],
        [0, mustache_height]
    ])
    dst_points = np.float32([
        [yaw_factor, pitch_factor],
        [mustache_width - yaw_factor, pitch_factor],
        [mustache_width + yaw_factor, mustache_height - pitch_factor],
        [-yaw_factor, mustache_height - pitch_factor]
    ])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_mustache = cv2.warpPerspective(rotated_mustache, perspective_matrix, (mustache_width, mustache_height))
    warped_alpha = cv2.warpPerspective(rotated_alpha, perspective_matrix, (mustache_width, mustache_height))

    # Overlay the warped mustache on the frame using alpha blending
    for c in range(3):  # For each color channel (RGB)
        frame[top_left_y:top_left_y+mustache_height, top_left_x:top_left_x+mustache_width, c] = \
            (warped_alpha * warped_mustache[:, :, c] + 
             (1 - warped_alpha) * frame[top_left_y:top_left_y+mustache_height, top_left_x:top_left_x+mustache_width, c])

    return frame


def get_frame_at(index, video):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)

    _, frame = video.read()

    return frame

def compute_coordinates(frame, face_mesh):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe's face mesh
    results = face_mesh.process(rgb_frame)

    # Ensure that at least one face is detected
    if not results.multi_face_landmarks:
        return None, None, None

    face_landmarks = results.multi_face_landmarks[0]

    # Get 3D coordinates for landmarks
    left_eye = face_landmarks.landmark[205]  # Left eye
    right_eye = face_landmarks.landmark[425]  # Right eye
    nose_tip = face_landmarks.landmark[1]  # Nose tip
    mouth_center = face_landmarks.landmark[13]  # Mouth center

    # Roll angle (rotation around Z-axis)
    dx_roll = right_eye.x - left_eye.x
    dz_roll = right_eye.z - left_eye.z
    roll_angle = np.degrees(np.arctan2(dz_roll, dx_roll))

    # Yaw angle (rotation around Y-axis)
    yaw_angle = np.degrees(np.arctan2(nose_tip.z, nose_tip.x))

    # Pitch angle (rotation around X-axis)
    dy_pitch = mouth_center.y - nose_tip.y
    dz_pitch = mouth_center.z - nose_tip.z
    pitch_angle = np.degrees(np.arctan2(dz_pitch, dy_pitch))

    # Reference coordinate for overlay placement
    coordinates = face_landmarks.landmark[164]  # Mouth or desired landmark

    return coordinates, -roll_angle, -yaw_angle, -pitch_angle



def smoothening_transition_interval(start_index, end_index, video, face_mesh, png_path):
    frames = []

    start_frame = get_frame_at(start_index, video)
    start_coords, start_roll_angle, start_yaw_angle, start_pitch_angle = compute_coordinates(start_frame, face_mesh)

    end_frame = get_frame_at(end_index, video)
    end_coords, end_roll_angle, end_yaw_angle, end_pitch_angle = compute_coordinates(end_frame, face_mesh)

    total_frames = end_index - start_index

    for curr_index in range(start_index, end_index):
        curr_x_rate = (end_coords.x - start_coords.x) / total_frames
        curr_y_rate = (end_coords.y - start_coords.y) / total_frames
        curr_roll_rate = (end_roll_angle - start_roll_angle) / total_frames
        curr_yaw_rate = (end_yaw_angle - start_yaw_angle) / total_frames
        curr_pitch_rate = (end_pitch_angle - start_pitch_angle) / total_frames

        off_set = curr_index - start_index

        curr_x = start_coords.x + off_set * curr_x_rate
        curr_y = start_coords.y + off_set * curr_y_rate
        curr_roll = start_roll_angle + off_set * curr_roll_rate
        curr_yaw = start_yaw_angle + off_set * curr_yaw_rate
        curr_pitch = start_pitch_angle + off_set * curr_pitch_rate

        png_coords = (curr_x, curr_y, curr_roll, curr_yaw, curr_pitch)

        frame = get_frame_at(curr_index, video)
        frame = print_png(frame, png_path, png_coords)

        frames.append(frame)

    return frames


def main():
    mustache_path = "mustache.png"

    video_path = "test_easy.mp4"
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = "output_with_mustache.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    smoothening_interval = 3

    try:
        for start_frame_index in range(0, total_frames, smoothening_interval):
            end_index = start_frame_index + smoothening_interval

            frames = smoothening_transition_interval(start_frame_index, end_index, cap, face_mesh, mustache_path)            

            for frame in frames:
                out.write(frame)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Processed video saved to {output_path}")

main()
