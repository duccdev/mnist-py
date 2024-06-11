import json
import signal
import numpy as np
from constants import *
from utils import read_to_string, write_to_file, one_hot


class NN:
    def __init__(self) -> None:
        self.weights = np.random.randn(OUTPUT_SIZE, INPUT_SIZE) * np.sqrt(
            2.0 / (OUTPUT_SIZE + INPUT_SIZE)
        )
        self.bias = np.zeros(OUTPUT_SIZE)
        self.epoch = 0

    def load(self, path: str) -> None:
        model = json.loads(read_to_string(path))
        self.weights = np.array(model["weights"])
        self.bias = np.array(model["bias"])
        self.epoch = model["epoch"]

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(
            -np.sum(y_true * np.log(np.clip(y_pred, 1e-12, 1.0)), axis=0)
        ) / len(y_true)

    def loss_deriv(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.clip(y_pred, 1e-12, 1.0) - y_true

    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.softmax((self.weights @ x) + self.bias)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
        epochs: int,
    ) -> None:
        stop_training = False

        def sigint_handler(signal, frame) -> None:
            global stop_training
            stop_training = True

        signal.signal(signal.SIGINT, sigint_handler)

        while True:
            total_loss = 0.0
            accuracy = 0.0

            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                loss = self.loss(one_hot(np.array([y]), OUTPUT_SIZE), output)
                accuracy += np.argmax(output) == y
                total_loss += loss

                grad = self.loss_deriv(one_hot(np.array([y]), OUTPUT_SIZE), output)
                grad_w = np.clip(np.outer(x, grad), -CLIP_VALUE, CLIP_VALUE)
                grad_b = np.clip(grad, -CLIP_VALUE, CLIP_VALUE).squeeze()

                self.weights -= learning_rate * grad_w.T
                self.bias -= learning_rate * grad_b

            total_loss /= len(x_train)
            accuracy = (accuracy + 1e-15) / len(x_train)
            self.epoch += 1

            print(
                f"\033[H\033[Jepoch: {self.epoch} | loss: {total_loss} | accuracy: {accuracy}"
            )

            if stop_training or self.epoch >= epochs:
                print("stopping training")
                break

    def save(self, path: str) -> None:
        write_to_file(
            path,
            json.dumps(
                {
                    "weights": self.weights.tolist(),
                    "bias": self.bias.tolist(),
                    "epoch": self.epoch,
                }
            ),
        )
