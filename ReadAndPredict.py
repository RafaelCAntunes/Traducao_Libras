import os
import numpy as np
import mediapipe as mp
import tensorflow as tf

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

#configurações mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistics = mp.solutions.holistic

holistics = mp_holistics.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=2)

#captura de vídeo
cap = cv2.VideoCapture(0)

# variaveis de controle

frame_sequence = []
predictions = []
threshold = 0.9

frames = 40 # Número de frames
acao = " "
frame_counter = 0
i = 0

#carregando modelo e ações
model = tf.keras.models.load_model(".venv/data/dataset/best_model.h5")
acoes = np.load(".venv/data/dataset/acoes.npy")



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

def draw_landmarks(image, results):

    # desenha os landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistics.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistics.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistics.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistics.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())

    return image

# Set mediapipe model
while cap.isOpened():

    ret, frame = cap.read()

    # Resize para o padrão
    frame = cv2.resize(frame, (640, 480))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps} frames per second")

    # faz a detecção dos landmarks
    image, results = mediapipe_detection(frame, holistics)
    print(results)

    # Desenho dos landmarks no frame e espelha a imagem
    image_with_landmarks = draw_landmarks(image, results)
    image_with_landmarks = cv2.flip(image_with_landmarks, 1)

    # coloca os landmarks em um vetor
    keypoints = extract_keypoints(results)
    frame_sequence.append(keypoints)
    frame_sequence = frame_sequence[-frames:]
    print(np.array(frame_sequence).shape)
    # action_to_read = frame_sequence[-frames:]

    if len(frame_sequence) == frames:  # quando o vetor tiver 40 landmarks de 40 frames diferentes
        # frame_sequence = action_to_read #garantindo que o vetor de frames não vá crescer indefinidamente, resetando sempre em 40.
        res = model.predict(np.expand_dims(frame_sequence, axis=0))[0]
        print(np.argmax(res))
        print(res[np.argmax(res)])
        print(acoes[np.argmax(res)])
        predictions.append(np.argmax(res))

        #  Logica de visualização
        # if np.unique(predictions[-15:])[0] == np.argmax(res): #Se as última predição dentre as 15 últimas predições é igual a atual
        if res[np.argmax(res)] > threshold:
            acao = acoes[np.argmax(res)]

    # escreve na tela a ação identificada
    cv2.putText(image_with_landmarks, ' '.join(acao), (3, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # as linhas a seguir são usadas apenas caso exista a necessidade de gravar os frames lidos
    # output_path = os.path.join(".venv/data/teste_captura", str(frame_counter).zfill(3)+".jpg")
    # cv2.imwrite(output_path, image_with_landmarks)
    # frame_counter += 1

    cv2.imshow('Tradutor Libras', image_with_landmarks)
    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()