import numpy as np, utils
from constants import *
from nn import NN


def main():
    datasets = utils.list_datasets()

    for dataset in datasets:
        print(f" - {dataset}")

    dataset = input("\nselect dataset: ")
    if dataset not in datasets:
        print("invalid dataset")
        exit(1)

    print("loading dataset")
    with np.load(f"datasets/{dataset}.npz", allow_pickle=True) as f:
        x_train = f["x_train"]
        y_train = f["y_train"]

        if x_train.ndim == 3 and x_train.shape[1] == 28 and x_train.shape[2] == 28:
            x_train = x_train.reshape(x_train.shape[0], 28 * 28) / 255.0
        else:
            print("dataset must be shaped like (n, 28, 28)")
            exit(1)

    models = utils.list_models()

    for model in models:
        print(f" - {model}")

    model_name = input(f"\nsave model with name ({dataset}): ") or dataset
    nn = NN()

    if model_name in models and input("train old model? (y/N): ") == "y":
        print("loading model")
        nn.load(f"models/{model_name}.json")

    try:
        learning_rate = float(input("learning rate: "))
    except ValueError:
        print("invalid learning rate")
        exit(1)

    try:
        epochs = int(input("epochs: "))
    except ValueError:
        print("invalid epochs")
        exit(1)

    print("training model")
    nn.train(x_train, y_train, learning_rate, epochs)

    print("saving model")
    nn.save(f"models/{model_name}.json")


if __name__ == "__main__":
    main()
