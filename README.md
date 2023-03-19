# EEG_Classification
A Classsification task performed on an EEG Dataset with the aid of Convolutional Neural Networks (CNNs) and Recurrent Neural Network (RNN) units.
The Code uses PyTorch.

Models include:
- Basic CNN consisting of CNN Layers and Fully connected layers.
- Maxpool CNN containing an additional maxpool layer.
- Temp CNN which additionally performs a 1D Convolution over the data.
- LSTM CNN which incorporates an RNN unit into the CNN model.
- MIx CNN which uses an RNN unit and a 1D Convolution layer.

All the models use Dropout to achieve better peformance.
Accuracies as high as 0.975 are observed.
