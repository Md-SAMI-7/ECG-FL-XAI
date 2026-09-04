import os
import torch
import torch.nn as nn
from models.resnet34_1d import ResNet1D34
from models.fedprox import compute_proximal_loss
from models.fedavg import federated_average

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
NUM_ROUNDS = 30
MU = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting FedProx Training (mu={MU}, {NUM_ROUNDS} Rounds) ---")
    global_model = ResNet1D34(in_channels=12, num_classes=12).to(DEVICE)
    print("Global Model initialized on:", DEVICE)

if __name__ == "__main__":
    main()
