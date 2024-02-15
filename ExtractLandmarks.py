import os
import cv2
import numpy as np
import mediapipe as mp

# Função para extrair e salvar landmarks em arquivo numpy que servirá como dataset

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

                    image, results = mediapipe_detection(image,holistics)

                    landmarks_array = extract_keypoints(results)

                    # Adiciona o array aos dados
                    landmarks_data.append(landmarks_array)

                interaction_data.append(landmarks_data)
                actions.append(actions_map[action])


            print(np.array(landmarks_array).shape)
            print(np.array(landmarks_data).shape)
            print(np.array(interaction_data).shape)


    # Cria o caminho  os arquivos numpy e salva os arrays
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "X"), interaction_data)
    np.save(os.path.join(output_folder, "Y"), actions)
    np.save(os.path.join(output_folder, "acoes"),  actions_list)

    # Libera os recursos do Mediapipe
    holistics.close()

# Exemplo de uso
if __name__ == "__main__":
    # Caminho para a pasta com os conjuntos de frames previamente trabalhados
    input_folder = ".venv/data/augmented"

    # Caminho para a pasta de saída para os landmarks
    output_folder = ".venv/data/dataset"

    # Extrai e salva os landmarks para cada ação e vídeo
    extract_and_save_landmarks(input_folder, output_folder)