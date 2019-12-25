import sys
import time
from pathlib import Path

import torch
from torch import cuda

def main(argv):
    start_time = time.time()

    # Model hyperparameters
    pos_parameters = {"hidden": 120, "batch_size": 10000, "lr": 0.01, "wd": 1e-5, "epochs": 20, "context": 5}
    ner_parameters = {"hidden": 40, "batch_size": 1000, "lr": 0.01, "wd": 1e-5, "epochs": 20, "context": 5}
    embedding_dim = 50

    if sys.argv[1] == 'pos':
        parameters = pos_parameters
    else:
        parameters = ner_parameters

    # Set gpu device
    # device = torch.device(argv[2] if torch.cuda.is_available() else 'cpu')

    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Graphical device test: {}".format(torch.cuda.is_available()))
    print("{} available".format(device))

    root = Path('data/{}'.format(argv[1]))

    # Path to data files
    train_file = root / "train"
    dev_file = root / "dev"
    test_file = root / "test"
    print(time.time() - start_time)


if __name__ == '__main__':
    main(sys.argv)