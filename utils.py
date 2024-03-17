import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd 

def save_plot(train_loss_list=[], valid_loss_list=[], test_loss_list=[]):
    if len(test_loss_list) == 0:
        plt.figure()
        red_patch = mpatches.Patch(color='red', label='Train Loss')
        blue_patch = mpatches.Patch(color='blue', label='Valid Loss')
        x_axis_data = list(range(1,len(train_loss_list)+1))
        sns.lineplot(x=x_axis_data, y=train_loss_list, color='red')
        sns.lineplot(x=x_axis_data, y=valid_loss_list, color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GNN Training Analysis")
        plt.legend(handles=[red_patch, blue_patch], loc='upper right')
        plt.savefig('./result/training_analysis.png')
        print("Saved ./result/training_analysis.png")
    else:
        plt.figure()
        y_real, y_pred = test_loss_list[0], test_loss_list[1]
        df = pd.DataFrame()
        df["gt_value"] = y_real
        df["pred_value"] = y_pred
        df["gt_value"] = df["gt_value"].apply(lambda row: row[0])
        df["pred_value"] = df["pred_value"].apply(lambda row: row[0])
        sns.scatterplot(data=df, x="gt_value", y="pred_value")
        plt.savefig('./result/testing_analysis.png')
        print("Saved ./result/testing_analysis.png")