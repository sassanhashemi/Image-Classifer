# Image Classifier

## Description
This image classsifier categorizes images from the CIFAR-10 dataset. Given an image, it predicts which of 10 possible classes the image belongs to using a PyTorch convolutional neural network. For more information on the CIFAR-10 dataset, visit https://www.cs.toronto.edu/~kriz/cifar.html.


## How to Use
1. Configure the `BATCH_SIZE` and `NUM_EPOCHS` variables in `constants.py`. You can also use the default values that are currently set.
2. Run `$ python3 train.py` to train the network. This can be run as many times as you'd like.
3. Run `$ python3 test.py` to test the performance of the network.

