import os
import cv2


# Função para extrair frames de um vídeo
def extract_frames(video_path, output_folder, start_frame=12, num_frames=40):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps} frames per second")

# Videos têm fps diferentes variando entre 24,29 e 59.

    if fps < 29:
        frame_skip = 0
        start_frame = 2
    elif fps <= 30:
        frame_skip = 1
    else:
        frame_skip = 4
        start_frame += 5

    print(f"Pulando {frame_skip} frames")
    print(f"Frame inicial:  {start_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(num_frames):

        for _ in range(frame_skip):
            success, _ = cap.read()
            if not success:
                break

        success, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        if not success:
            break

        frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)

    cap.release()


# Caminho para a pasta com os vídeos
video_folder = ".venv/data/videos"

# Itera pelos vídeos na pasta
for action in os.listdir(video_folder):
    folder_path = os.path.join(video_folder, action)
    contador = 0
    for video_name in os.listdir(folder_path):
        contador += 1
        if video_name.endswith(".mp4"):  # ou o formato do seu vídeo
            video_path = os.path.join(folder_path, video_name)
            video_folder_name = str(contador).zfill(3)  # Remove a extensão do vídeo

            # Cria o caminho para a nova pasta
            new_folder_path = os.path.join('.venv/data/frames', action, video_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            # Extrai 40 frames, começando do 12º frame, para cada vídeo
            extract_frames(video_path, new_folder_path, start_frame=12, num_frames=40)

            print(f"Processando vídeo: {video_name}")
            print(f"Caminho da nova pasta: {new_folder_path}")
