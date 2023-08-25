import torch
import torchvision
from Network import Network
from constants import *


if __name__ == "__main__":

    # Download and normalize testing data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testing_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    
    # Load the network
    network = Network()
    try:
        network.load_state_dict(torch.load(NETWORK_PATH))
    except:
        print("Error: No network has been saved. Run 'python3 train.py'")
        exit(1)


    # Measure performance by class
    correct_by_class = {classname: 0 for classname in CLASSES}
    total_by_class = {classname: 0 for classname in CLASSES}

    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            outputs = network(images)
            _, predictions = torch.max(outputs.data, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_by_class[CLASSES[label]] += 1
                total_by_class[CLASSES[label]] += 1

    for classname, correct_count in correct_by_class.items():
        print("Accuracy for class {} is {}%".format(classname, 100*correct_count//total_by_class[classname]))

 
    # Measure total performance
    correct = sum(correct_by_class.values())
    total = sum(total_by_class.values())
    print("Accuracy for all classes is {}%".format(100*correct//total))



