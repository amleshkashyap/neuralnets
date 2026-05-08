import tensorflow as tf
import numpy as np

class Preprocess:
    def __init__(self, text: str, seq_length):
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []
        self.chars = sorted(list(set(text)))
        char_to_index = { char: i for i, char in enumerate(self.chars) }
        index_to_char = { i: char for i, char in enumerate(self.chars) }

        for i in range(len(text) - self.seq_length):
            seq = text[i:i + self.seq_length]
            label = text[i + self.seq_length]
            self.sequences.append([char_to_index[char] for char in seq])
            self.labels.append(char_to_index[label])

        self.X = np.array(self.sequences)
        self.y = np.array(self.labels)

        self.X_one_hot = tf.one_hot(self.X, len(self.chars))
        self.y_one_hot = tf.one_hot(self.y, len(self.chars))

    def get_chars(self):
        return self.chars

    def get_x_one_hot(self):
        return self.X_one_hot

    def get_y_one_hot(self):
        return self.y_one_hot