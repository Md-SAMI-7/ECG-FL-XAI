import os
import torch
import torch.nn as nn
from models.resnet34_1d import ResNet1D34
from models.fedavg import federated_average

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting FedAvg Training ({NUM_ROUNDS} Communication Rounds) ---")
    global_model = ResNet1D34(in_channels=12, num_classes=12).to(DEVICE)
    print("Global Model initialized on:", DEVICE)
    print("Clients:", FEDERATED_CLIENTS)

if __name__ == "__main__":
    main()
