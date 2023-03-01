import torch
from torch import nn
from trainer import *



class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(.2),
            torch.nn.BatchNorm2d(128),
        )

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        self.num_output_features = 128*2*2 #sol

        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )
        apply_weight_init(self)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # Run image through convolutional layers

        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    early_stop_count = 5
    filename = "TTK4852/df_all.csv"
    data_all = pd.read_csv(filename)
    data_train, data_test = train_test_split(data_all, test_size=0.1, random_state=42)

    train_array = convert_to_numpy(data_train)
    test_array= convert_to_numpy(data_test)

    dataloaders = load_tare(batch_size,train_array, test_array)
    model = ExampleModel(image_channels=66, num_classes=2)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    trainer.train()
    create_plots(trainer, "Learningrate")
    print_accuracy(trainer)