# Spectrogram Voice Analysis with ResNet-152

This repository contains code for a deep learning project aimed at analyzing voice data using spectrograms and a ResNet-152 architecture. The project focuses on classifying audio samples into different categories based on the UrbanSound8K dataset.

## Dataset

The dataset used in this project is the UrbanSound8K dataset, available on Kaggle. It consists of 8732 labeled sound excerpts (4 seconds each) of urban sounds across 10 classes. You can find the dataset [here](https://www.kaggle.com/datasets/chrisfilo/urbansound8k).

## Usage

1. **Data Preparation**: Download the UrbanSound8K dataset from the provided link. Extract the dataset and ensure it is structured correctly. The dataset includes metadata in CSV format along with audio files categorized into different folds.

2. **Environment Setup**: This project was developed using Python in a Jupyter Notebook environment. Ensure you have the necessary libraries installed, including `pandas`, `numpy`, `matplotlib`, `seaborn`, `librosa`, `scikit-learn`, `tensorflow`, `opencv-python`, and `resampy`.

3. **Notebook Execution**: Run the Jupyter Notebook [spectrogram-voice-analysis-with-ResNet-152.ipynb](https://github.com/elaaatif/spectrogram-voice-analysis-with-ResNet-152/blob/main/spectrogram-voice-analysis-with-ResNet-152.ipynb) to execute the project. This notebook guides you through the steps of data loading, feature extraction, model building, training, evaluation, and visualization.

4. **Model Usage**: After training, the trained model is saved as [ResNet152_based_model.zip](https://github.com/elaaatif/spectrogram-voice-analysis-with-ResNet-152/blob/main/ResNet152_based_model.zip). You can extract and load this model in your Python code using TensorFlow/Keras and use it for inference on new audio samples.

    ```python
    from tensorflow.keras.models import load_model

    # Load the trained model
    model = load_model('ResNet152_based_model.h5')

    # Perform inference on new data
    # Replace X_new with your new data
    predictions = model.predict(X_new)
    ```

## Model Architecture

The model architecture consists of a pre-trained ResNet-152 base with custom layers added on top. The ResNet-152 base is frozen to leverage the learned features from ImageNet, while custom layers adapt the model for spectrogram-based voice analysis. The architecture includes:

### Model Architecture: ResNet-152 with Custom Layers

1. **Base Model**: The base of the model is ResNet-152, a deep convolutional neural network (CNN) architecture known for its effectiveness in image classification tasks. ResNet-152 consists of 152 layers and has shown impressive performance on various computer vision tasks. You utilize the pre-trained weights from ImageNet to leverage the learned features.

2. **Freezing Base Layers**: All layers of the ResNet-152 base model are frozen, meaning they are not updated during training. This approach allows the model to retain the learned representations from ImageNet while fine-tuning the model's parameters for the specific task of sound classification.

3. **Custom Layers**: On top of the frozen ResNet-152 base, custom layers are added to adapt the model for spectrogram-based voice analysis:

    - **Global Average Pooling 2D (GAP)**: After the base model, a Global Average Pooling layer is added to reduce the spatial dimensions of the feature maps to a vector of features. This helps in capturing the most important features from the spectrogram.

    - **Dense Layers**: Multiple fully connected Dense layers are added to the model to learn complex patterns from the spectrogram features. The Dense layers use Rectified Linear Unit (ReLU) activation functions to introduce non-linearity into the model.

    - **Dropout Regularization**: Dropout layers are added after some Dense layers to prevent overfitting. Dropout randomly sets a fraction of input units to zero during training, which helps in improving the generalization of the model.

    - **Output Layer**: The final Dense layer predicts the probability distribution over the classes using a softmax activation function. The number of units in this layer corresponds to the number of classes in the dataset.

### Summary

The model architecture combines the power of a pre-trained ResNet-152 base with custom layers tailored for spectrogram-based voice analysis. By leveraging transfer learning and fine-tuning, the model can effectively classify audio samples into different categories. The addition of custom layers allows the model to adapt to the specific characteristics of the spectrogram data while maintaining the robustness and representational power of the pre-trained ResNet-152 base.
