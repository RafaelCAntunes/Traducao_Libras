from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score


#carrega as variáveis
X = np.load(".venv/data/dataset/X.npy")
Y = np.load(".venv/data/dataset/Y.npy")
acoes = np.load(".venv/data/dataset/acoes.npy")
print(acoes)

print(X.shape)
print(Y.shape)
print(Y)


#transforma Y em categórico
Y_categorical = to_categorical(Y).astype(int)
print(Y_categorical.shape)
print(Y_categorical[0])

#divide X e Y em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, Y_categorical, test_size=0.1)

print(X_train.shape)

#cria o callback de log e de modelcheckpoint para salvar o modelo conforme a accuracy melhora
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='categorical_accuracy', save_best_only=True)]


#Cria a Rede Neural
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(acoes.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#Treina a Rede
model.fit(X_train, y_train, epochs=200, callbacks=[callbacks, tb_callback])


#carrega o melhor modelo salvo
model = tf.keras.models.load_model(".venv/data/dataset/best_model.h5")


#Testa o predic com o X de teste
res = model.predict(X_test)

print(acoes[np.argmax(res[4])])
print(acoes[np.argmax(y_test[4])])

# verifica acuracia
y_predito = model.predict(X_test)
y_real = np.argmax(y_real, axis=1).tolist()
y_predito = np.argmax(y_predito, axis=1).tolist()

print(accuracy_score(y_real, y_predito))
