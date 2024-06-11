import numpy as np, os


def read_to_string(path: str) -> str:
    with open(path) as fp:
        return fp.read()


def write_to_file(path: str, content: str) -> None:
    with open(path, "w") as fp:
        fp.write(content)


def one_hot(values: np.ndarray, labels: int) -> np.ndarray:
    return np.eye(labels)[values]


def list_datasets() -> list[str]:
    return [
        "".join(path.rsplit(".npz", maxsplit=1))
        for path in os.listdir("datasets")
        if path.endswith(".npz")
    ]


def list_models() -> list[str]:
    return [
        "".join(path.rsplit(".json", maxsplit=1))
        for path in os.listdir("models")
        if path.endswith(".json")
    ]
