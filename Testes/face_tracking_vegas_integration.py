from deepface import DeepFace
from moviepy.editor import VideoFileClip

last_valid_pos = (0, 0, 1920, 1080)

def compute_face_pos(frame):
    global last_valid_pos

    h, w = frame.shape[:2]

    detections = DeepFace.extract_faces(img_path=frame, detector_backend='yolov8', enforce_detection=False, align=True)

    if len(detections) == 0:
        return last_valid_pos

    detection = detections[0] 

    if detection['confidence'] < 0.6:
        return last_valid_pos

    x, y, w, h = detection["facial_area"]['x'], detection["facial_area"]['y'], detection["facial_area"]['w'], detection["facial_area"]['h']
    last_valid_pos = (x,y,w,h)

    return last_valid_pos

def get_frame_at_index(frame_index, video):
    time = frame_index / video.fps
    frame = video.get_frame(time)
    return frame

def get_total_frames(video):
    duration = video.duration
    fps = video.fps
    total_frames = int(duration * fps)
    return total_frames

def process_video(input_path, output_path, progress_callback=None):
    video = VideoFileClip(input_path)
    frame_interval = int(video.fps)
    total_frames = get_total_frames(video)

    with open(output_path, 'w') as file:
        for frame_index in range(frame_interval*1607, total_frames, frame_interval): #AJUSTAR ISSO AQUI PELO AMOR DE DEWUS
            curr_pos = compute_face_pos(get_frame_at_index(frame_index, video))
            file.write(f"{curr_pos}\n")
            
            print(f'{(frame_index / total_frames * 100):.0f}%')

            if progress_callback:
                progress_callback(frame_index + frame_interval, total_frames)

def main():
    input_video_path = '2024-08-30 16-09-08.mp4'
    output_video_path = f"{input_video_path}_coords_continuation.txt"
    process_video(input_video_path, output_video_path)
