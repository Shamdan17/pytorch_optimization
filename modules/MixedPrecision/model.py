from torchvision.models import resnet18


def get_model():
    # Create model
    model = resnet18(num_classes=10)

    return model
