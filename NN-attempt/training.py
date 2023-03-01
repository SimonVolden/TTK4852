import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from trainer import Trainer
from trainer import compute_loss_and_accuracy 


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    latex_figures = pathlib.Path("../latex/figures") 
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.savefig(latex_figures.joinpath(f"{name}_plot.png")) 
    plt.show()

def print_accuracy(trainer: Trainer): 
    datasets = { 
        "train": trainer.dataloader_train,
        "test": trainer.dataloader_test,
        "val": trainer.dataloader_val
    }
    trainer.load_best_model() 
    for dset, dl in datasets.items():
        avg_loss, accuracy = compute_loss_and_accuracy(dl, trainer.model, trainer.loss_criterion)
        print(
            f"Dataset: {dset}, Accuracy: {accuracy}, loss: {avg_loss}"
        )
