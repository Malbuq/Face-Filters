from deepface import DeepFace
import cv2
from moviepy.editor import VideoFileClip


last_valid_pos = (0,0,1920,1080)

def compute_face_pos(frame):

    h, w = frame.shape[:2]

    detection = DeepFace.extract_faces(img_path=frame, detector_backend='yolov8', enforce_detection=False, align=True)[0]

    if detection['confidence'] < 0.6:
        return None

    x, y, w, h = detection["facial_area"]['x'], detection["facial_area"]['y'], detection["facial_area"]['w'], detection["facial_area"]['h']

    return (x, y, w, h)

def build_square_frame(frame, face_pos):
    h, w = frame.shape[:2]
    face_w = face_pos[2]
    face_h = face_pos[3]

    face_center_x = face_pos[0] + face_w // 2
    face_center_y = face_pos[1] + face_h // 2

    square_size = 1000

    start_x = face_center_x - square_size // 2
    start_y = face_center_y - square_size // 2

    if start_x + square_size > w:
        start_x = w - square_size
    if start_y + square_size > h:
        start_y = h - square_size

    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0

    cropped_frame = frame[start_y:start_y + square_size, start_x:start_x + square_size]

    return cropped_frame

def get_frame_at_index(frame_index, video):
    time = frame_index / video.fps
    frame = video.get_frame(time)
    return frame

def get_total_frames(video):
    duration = video.duration
    fps = video.fps
    total_frames = int(duration * fps)
    return total_frames

def smoothening_transition_interval(start_frame_index, end_frame_index, video):
    frames = []

    global last_valid_pos

    start_frame = get_frame_at_index(start_frame_index, video)
    start_face_pos = compute_face_pos(start_frame)

    if start_face_pos is None:
        start_face_pos = last_valid_pos
    
    last_valid_pos = start_face_pos

    end_frame = get_frame_at_index(end_frame_index, video)
    end_frame_pos = compute_face_pos(end_frame)

    if end_frame_pos is None:
        end_frame_pos = last_valid_pos

    frame_interval = end_frame_index - start_frame_index

    for curr_frame_index in range(start_frame_index, end_frame_index):
        curr_frame = get_frame_at_index(curr_frame_index, video)

        curr_x_rate = (end_frame_pos[0] - start_face_pos[0]) / frame_interval
        curr_y_rate = (end_frame_pos[1] - start_face_pos[1]) / frame_interval
        curr_w_rate = (end_frame_pos[2] - start_face_pos[2]) / frame_interval
        curr_h_rate = (end_frame_pos[3] - start_face_pos[3]) / frame_interval

        off_set = curr_frame_index - start_frame_index

        curr_x = start_face_pos[0] + off_set * curr_x_rate
        curr_y = start_face_pos[1] + off_set * curr_y_rate
        curr_w = start_face_pos[2] + off_set * curr_w_rate
        curr_h = start_face_pos[3] + off_set * curr_h_rate

        curr_face_pos = (int(round(curr_x)), int(round(curr_y)), int(round(curr_w)), int(round(curr_h)))

        output_frame = build_square_frame(curr_frame, curr_face_pos)
        frames.append(output_frame)

    return frames

def process_video(input_path, output_path, frame_interval):
    video = VideoFileClip(input_path)
    total_frames = get_total_frames(video)

    output_clip = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1000, 1000))

    for start_frame_index in range(0, total_frames, frame_interval):
        end_frame_index = start_frame_index + frame_interval

        if start_frame_index + frame_interval > total_frames:
            end_frame_index = total_frames

        print(f'{(start_frame_index / total_frames * 100):.0f}%')

        frames = smoothening_transition_interval(start_frame_index, end_frame_index, video)
        
        for frame in frames:
            output_clip.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    output_clip.release()
    video.reader.close()

def main():
    input_video_paths = ['untitled.mp4','1T9A5098.mp4', '1T9A5099.mp4']
    for path in input_video_paths:
        process_video(path, f'{path}-(1x1).mp4', 60)
main()