import os
import random

import cv2
import imgaug.augmenters as iaa


# Função para aplicar data augmentation usando imgaug
def apply_augmentation(input_folder, output_folder, num_augmentations=5):

    for action in os.listdir(input_folder):
        action_path = os.path.join(input_folder, action)
        contador = 0
        for articulator in os.listdir(action_path):
            articulator_path = os.path.join(action_path,articulator)

            if os.path.isdir(articulator_path):
                images = [cv2.imread(os.path.join(articulator_path, f"frame_{i}.jpg")) for i in range(40)]

                # Aplica data augmentation múltiplas vezes
                for augmentation_num in range(num_augmentations):
                    contador += 1
                    flip = random.randint(0, 1)
                    brightness_factor = random.uniform(0.7, 1.3)
                    saturation_factor = random.uniform(0.5, 1.5)
                    shift_value = random.randint(-320, 320)
                    #zoom_factor = random.uniform(0.5, 2.0)

                    flip_lr = iaa.Fliplr(flip)
                    multiply_brightness = iaa.Multiply(brightness_factor)
                    multiply_saturation = iaa.MultiplySaturation(saturation_factor)
                    affine = iaa.Affine(translate_px={"x": shift_value})


                    # Cria uma sequência determinística para garantir consistência nos valores
                    seq = iaa.Sequential([
                        flip_lr,
                        multiply_brightness,
                        multiply_saturation,
                        affine
                        # Adicione outras augmentations conforme necessário
                    ], random_order=False).to_deterministic()

                    # Aplica a sequência determinística a todos os frames
                    augmented_images = seq(images=images)

                    # Cria uma nova pasta numerada para armazenar o conjunto augmentado
                    new_output_folder = os.path.join(output_folder, action, str(contador).zfill(3))
                    os.makedirs(new_output_folder, exist_ok=True)

                    # Salva os frames augmentados
                    for i, augmented_image in enumerate(augmented_images):
                        output_path = os.path.join(new_output_folder, str(i+1).zfill(3)+".jpg")
                        cv2.imwrite(output_path, augmented_image)


if __name__ == "__main__":
    # Caminho para a pasta com os conjuntos de frames originais
    input_folder = ".venv/data/Frames"

    # Caminho para a pasta de saída para os conjuntos augmentados
    output_folder = ".venv/data/augmented"

    # Aplica data augmentation 15 vezes para cada conjunto de 40 frames
    apply_augmentation(input_folder, output_folder, num_augmentations=15)
