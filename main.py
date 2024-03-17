import argparse
import torch
from tqdm import tqdm
import os
import glob
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
from gnn_model import GCN
from utils import save_plot

parser = argparse.ArgumentParser(description = "Graph Neural Networks for estimating water solubility of a molecule structure.")
parser.add_argument('-lr', '--learning_rate', default = 4e-3)
parser.add_argument('-ep', '--epoch', default = 2000)
parser.add_argument('-m', '--mode', default="train")
parser.add_argument('-g', '--num_graphs_per_batch', default=6)
args = parser.parse_args()

lr = args.learning_rate
total_epoch = int(args.epoch)
MODE = args.mode.lower()
num_graphs_per_batch = int(args.num_graphs_per_batch)

data = MoleculeNet(root="./dataset/",name="ESOL")
num_features = data.num_features # features of a node

gcn_model = GCN(num_features)

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(gcn_model.parameters(), lr) 
lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.05)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = gcn_model.to(device)


data_size = len(data)

train_loader = DataLoader(data[:int(data_size * 0.8)], batch_size=num_graphs_per_batch, shuffle=True)
valid_loader = DataLoader(data[int(data_size * 0.8):int(data_size * 0.9)], batch_size=num_graphs_per_batch, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.9):], batch_size=num_graphs_per_batch, shuffle=True)


def run_training():
    model.train()
    # Enumerate over the data
    for batch in train_loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      # Calculating the loss and gradients
      loss = loss_fn(pred, batch.y)     
      loss.backward()  
      # Update using the gradients
      optimizer.step()   
    return loss, embedding


def run_validation():
    model.eval()
    # Enumerate over the data
    for batch in valid_loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      with torch.no_grad():
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch.y)     
    return loss, embedding

def run_testing():
    model.eval()
    y_real_list, y_pred_list = [], []
    for batch in test_loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      with torch.no_grad():
        y_real_list.extend(batch.y.tolist())
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)  
        y_pred_list.extend(pred.detach().tolist())    
    return y_real_list, y_pred_list

if MODE == "train":
    print("Starting training...")
    train_losses, valid_losses = [], []
    saved_validation_loss = 1000000
    for epoch in tqdm(range(1, total_epoch+1), desc= "Training Epoch"):
        train_loss, h = run_training()
        lr_scheduler.step()
        valid_loss, valid_h = run_validation()
        train_loss = train_loss.detach().numpy()
        valid_loss = valid_loss.detach().numpy()
        train_losses.append(np.float32(train_loss))
        valid_losses.append(np.float32(valid_loss))
        # if epoch % 5 == 0:
            # if valid_loss < saved_validation_loss:
            #     saved_validation_loss = valid_loss
            #     os.system("rm ./weights/*.pt")
            #     torch.save(model.state_dict(),"./weights/"+str(epoch)+".pt")
            #     print("Weight saved at epoch: ", epoch)
            # print(f"Epoch {epoch} | Train Loss {train_loss} | Valid Loss {valid_loss}")
        os.system("rm ./weights/*.pt")
        torch.save(model.state_dict(),"./weights/"+str(epoch)+".pt")
        print("Weight saved at epoch: ", epoch)           
        print(f"Epoch {epoch} | Train Loss {train_loss} | Valid Loss {valid_loss}")
    save_plot(train_loss_list=train_losses, valid_loss_list=valid_losses)


if MODE in ["test", "train"]:
    weight_filename_list = glob.glob("./weights/*.pt")
    if len(weight_filename_list) !=1:
        print("Error: Weight file not present inside ./weights/")
        exit()
    else:
        weight_filename = weight_filename_list[0]
        print("Loading weight file for testing: ", weight_filename)
        model.load_state_dict(torch.load(weight_filename))
        y_real_list, y_pred_list = run_testing()
        save_plot(test_loss_list=[y_real_list, y_pred_list])

print("Code Executed Successfully.")
