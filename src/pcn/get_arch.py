from torch import nn


# Use proper initialization for Linear
class MyLinear(nn.Linear):
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # nn.init.xavier_uniform_(self.weight, gain)
        nn.init.orthogonal_(self.weight, gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# Define class predictor (Sigmoid for MSE vs. nothing for CrossEntropy)
def class_predictor(dim_in: int, dim_out: int, use_CELoss: bool):
    return (
        MyLinear(dim_in, dim_out)  # for CrossEntropy
        if use_CELoss
        else nn.Sequential(MyLinear(128, 10), nn.Sigmoid())  # for MSE
    )


def get_architecture_bpc(dataset: str, model_name: str = None, activation="gelu"):
    if dataset == "CIFAR10":
        num_classes = 10
        img_size = 32
    elif dataset == "CIFAR100":
        num_classes = 100
        img_size = 32
    elif dataset == "tiny-imagenet":
        num_classes = 200
        img_size = 64

    
    if activation == "relu":
        activation = nn.ReLU
    elif activation == "gelu":
        activation = nn.GELU
    elif activation == "tanh":
        activation = nn.Tanh
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU
    else:
        raise ValueError("Unsupported activation function. Only relu, gelu, tanh and leaky_relu are supported.")



    if dataset == "EMNIST" or dataset == "FashionMNIST":
        architecture_up = [
            nn.Sequential(MyLinear(28 * 28, 128), activation()),
            nn.Sequential(MyLinear(128, 128), activation()),
            nn.Sequential(MyLinear(128, 128), activation()),
            nn.Sequential(MyLinear(128, 10)),
        ]

        architecture_down = [
            nn.Sequential(MyLinear(10, 128,), activation()),
            nn.Sequential(MyLinear(128, 128), activation()),
            nn.Sequential(MyLinear(128, 128), activation()),
            nn.Sequential(MyLinear(128, 28 * 28), nn.Tanh()),
        ]
    elif dataset in ("CIFAR10", "CIFAR100", "tiny-imagenet"):
        if model_name == "VGG5":
            architecture_up = [
                nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**4)**2 , num_classes, bias=True)),
            ]
            architecture_down = [
                nn.Sequential(nn.Linear(num_classes, 512 * (img_size // 2**4)**2, bias=True), nn.Unflatten(1, (512, img_size // 2**4, img_size // 2**4)), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(128, 3, 3, 2, 1, 1), nn.Tanh()),  # Output layer with Tanh
            ]
        elif model_name == "VGG9": 
            architecture_up = [
                nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1), nn.BatchNorm2d(128), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), activation()),
                nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), activation()),
                nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation()),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation()),
                nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**4)**2, num_classes, bias=True)),
            ]
            architecture_down = [
                nn.Sequential(nn.Linear(num_classes, 512 * (img_size // 2**4)**2, bias=True), nn.Unflatten(1, (512, img_size // 2**4, img_size // 2**4)), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 1, 1, 0), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 1, 1, 0), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(256, 256, 3, 1, 1, 0), activation()),
                nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(128, 128, 3, 1, 1, 0), activation()),
                nn.Sequential(nn.ConvTranspose2d(128, 3, 3, 2, 1, 1), nn.Tanh()),  # Output layer with Tanh
            ]
        elif model_name == "VGG16":
            architecture_up = [
                nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), activation()),
                nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), activation()),
                nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), activation()),
                nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), activation()),
                nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), activation()),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation()),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation()),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation()),
                nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), activation(), nn.MaxPool2d(2, 2)),
                nn.Sequential(nn.Flatten(), nn.Linear(512 * (img_size // 2**5)**2 , num_classes)),
            ]
            architecture_down = [
                nn.Sequential(nn.Linear(num_classes, 512 * (img_size // 2**5)**2), nn.Unflatten(1, (512, img_size // 2**5 , img_size // 2**5)), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 512, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(512, 256, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(256, 256, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(256, 256, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(128, 128, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), activation()),
                nn.Sequential(nn.ConvTranspose2d(64, 3, 3, 1, 1), nn.Tanh()),  # Output layer with Tanh
            ]

        else:
            raise ValueError("model is not supported.")
    return [architecture_up, architecture_down]

