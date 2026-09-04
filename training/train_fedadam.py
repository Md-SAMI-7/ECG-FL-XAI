import os
import torch
from models.resnet34_1d import ResNet1D34
from models.fedadam import FedAdamServerOptimizer
from models.fedavg import federated_average

FEDERATED_CLIENTS = ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo", "ptb-xl"]
NUM_ROUNDS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- Starting FedAdam Training ({NUM_ROUNDS} Rounds) ---")
    global_model = ResNet1D34(in_channels=12, num_classes=12).to(DEVICE)
    server_opt = FedAdamServerOptimizer(global_model.state_dict())
    print("Global Model & Adaptive Server Optimizer initialized on:", DEVICE)

if __name__ == "__main__":
    main()
