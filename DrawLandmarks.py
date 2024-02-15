import os
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh,  rh])
def extract_and_save_landmarks(input_folder, output_folder):
    # Inicializa o módulo Holistics do Mediapipe
    mp_holistics = mp.solutions.holistic
    holistics = mp_holistics.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=2)
    interaction_data = []
    actions = []
    diretorios = [acao for acao in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, acao))]
    actions_list = np.array([os.path.basename(diretorio) for diretorio in diretorios])
    actions_map = {acao: num for num, acao in enumerate(diretorios)}
    for action in os.listdir(input_folder):
        action_path = os.path.join(input_folder, action)
        print(os.listdir(action_path))
        for iteration in os.listdir(action_path):
            iteration_path = os.path.join(action_path, iteration)

            if os.path.isdir(iteration_path):
                landmarks_data = []
                print("frames:")
                print(os.listdir(iteration_path))
                for frame in os.listdir(iteration_path):
                    frame_path = os.path.join(iteration_path, frame)
                    image = cv2.imread(frame_path)

                    image, results = mediapipe_detection(image, holistics)

                    image_with_landmarks = draw_landmarks(image,results)

                    actual_output = os.path.join(output_folder,action,iteration)

                    os.makedirs(actual_output, exist_ok=True)
                    output_path = os.path.join(actual_output, frame)
                    cv2.imwrite(output_path, image_with_landmarks)




    # Libera os recursos do Mediapipe
    holistics.close()


def draw_landmarks(image, results):

    # desenha os landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())

    return image

# Exemplo de uso
if __name__ == "__main__":
    # Caminho para a pasta com os conjuntos de frames previamente trabalhados
    input_folder = ".venv/data/augmented"

    # Caminho para a pasta de saída para os landmarks
    ##output_folder = ".venv/data/landmarks"
    output_folder = ".venv/data/landmarks_sem_rosto"

    # Extrai e salva os landmarks para cada ação e vídeo
    extract_and_save_landmarks(input_folder, output_folder)