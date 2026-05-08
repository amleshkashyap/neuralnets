from Preprocess import Preprocess
from RecurrentNN import RecurrentNN


if __name__ == "__main__":
    preprocess = Preprocess("This is Gesaffelstein, a popular DJ and music producer from France", 3)
    rnn = RecurrentNN(3, preprocess.get_chars())

    rnn.compile_model()
    rnn.train(preprocess.get_x_one_hot(), preprocess.get_y_one_hot(), 100)

    rnn.generate_new_text()