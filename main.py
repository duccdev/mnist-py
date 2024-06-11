import numpy as np, cv2, utils, signal
from nn import NN

signal.signal(signal.SIGINT, (lambda s, f: exit(0)))


def main():
    models = utils.list_models()

    for model in models:
        print(f" - {model}")

    model = input("\nload model: ")
    if model not in models:
        print("invalid model")

    print("loading model")
    nn = NN()
    nn.load(f"models/{model}.json")

    use_samples = input("use samples from dataset? (y/N): ") == "y"
    show_sample = False

    if use_samples:
        datasets = utils.list_datasets()

        for dataset in datasets:
            print(f" - {dataset}")

        dataset = input("\nselect dataset: ")
        if dataset not in datasets:
            print("invalid dataset")

        print("loading dataset")

        with np.load(f"datasets/{dataset}.npz", allow_pickle=True) as f:
            og_x_train = f["x_train"]
            x_train = f["x_train"]
            y_train = f["y_train"]

            if x_train.ndim == 3 and x_train.shape[1] == 28 and x_train.shape[2] == 28:
                x_train = x_train.reshape(x_train.shape[0], 28 * 28) / 255.0
            else:
                print("dataset must be shaped like (n, 28, 28)")
                exit(1)

        show_sample = input("always show/save sample to sample.png? (y/N): ") == "y"

    while True:
        if not use_samples:
            path = input("path to image (ctrl+c to exit): ")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("invalid path")
                continue

            if img.shape[0] < 28 or img.shape[1] < 28:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            elif img.shape[0] > 28 or img.shape[1] > 28:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            img = np.array(img, dtype=np.float32).reshape(28 * 28) / 255.0
        else:
            sample = input(
                f"select sample (1 to {x_train.shape[0]}) (ctrl+c to exit): "
            )

            try:
                sample = int(sample) - 1
            except ValueError:
                print("invalid sample")
                continue

            if sample < 0 or sample >= 60000:
                print("invalid sample")
                continue

            img = x_train[sample]
            print(f"sample: {y_train[sample]}")

            if show_sample:
                cv2.imwrite("sample.png", og_x_train[sample].reshape(28, 28))

        pred = nn.predict(img)

        print(f"raw prediction: {pred}")
        print(
            f"prediction: {np.argmax(pred)} with {np.max(pred) * 100:.2f}% confidence"
        )


if __name__ == "__main__":
    main()
