import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

class RecurrentNN:
    def __init__(self, seq_length, chars: list):
        self.seq_length = seq_length
        self.chars = chars

        self.model = Sequential()
        self.model.add(SimpleRNN(50, input_shape = (self.seq_length, len(self.chars)), activation = 'relu'))
        self.model.add(Dense(len(self.chars), activation = 'softmax'))

    def compile_model(self):
        self.model.compile(optimizer = 'adam',
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])

    def get_model(self):
        return self.model

    def train(self, x_one_hot, y_one_hot, epochs = 100):
        self.model.fit(x_one_hot, y_one_hot, epochs = epochs)

    def generate_new_text(self, start_sequence = "This is G"):
        generated_text = start_sequence
        char_to_index = { char: i for i, char in enumerate(self.chars) }
        index_to_char = { i: char for i, char in enumerate(self.chars) }

        for i in range(50):
            x = np.array([[char_to_index[char] for char in generated_text[-self.seq_length:]]])
            x_one_hot = tf.one_hot(x, len(self.chars))
            prediction = self.model.predict(x_one_hot)
            next_index = np.argmax(prediction)
            next_char = index_to_char[next_index]
            generated_text += next_char

        print("Generated Text: ")
        print(generated_text)