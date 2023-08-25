import torch
import torchvision
from Network import Network
from constants import *

if __name__ == '__main__':

    # Download and normalize training data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    # Define network, loss function, and optimizer
    network = Network()
    try:
        network.load_state_dict(torch.load(NETWORK_PATH))
        print("Loaded saved network")
    except:
        print("Couldn't find saved network, starting with new network")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)


    # Train the network
    print("Starting training")
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(training_loader): # start=0 is default value
            inputs, labels = data
            optimizer.zero_grad()
            
            # forward, backward, optimize
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statisitics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[epoch={}, iter={}] loss: {}".format(epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    print("Finished training")


    # Save trained model
    torch.save(network.state_dict(), NETWORK_PATH)
