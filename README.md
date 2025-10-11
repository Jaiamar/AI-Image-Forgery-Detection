AI-Powered Image Forgery Detection

This project utilizes a Convolutional Neural Network (CNN) to detect forged or manipulated images. The model is trained to classify images as either "real" or "fake".

Description

The goal of this project is to build and train a deep learning model capable of identifying doctored images. By leveraging a CNN, the model learns to recognize patterns and artifacts that are often introduced during the image manipulation process.

Features

**CNN-based classification model** for forgery detection.
**Training and prediction scripts** to easily train the model and test it on new images.
**Early stopping** mechanism to prevent overfitting and save the best model.
**Learning rate scheduling** to adjust the learning rate during training for better convergence.

Dataset

[](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)

Installation

1.Clone the repository:

    ```bash
    git clone https://github.com/Jaiamar/AI-Image-Forgery-Detection.git
    ```

2.Navigate to the project directory:

    ```bash
    cd AI-Image-Forgery-Detection
    ```

3.Install the required dependencies:
    It is recommended to create a `requirements.txt` file with all the necessary libraries.

    ```bash
    pip install -r requirements.txt
    ```

    A basic `requirements.txt` file might include:

    ```
    torch
    torchvision
    numpy
    Pillow
    matplotlib
    ```

Usage

Training the Model

To train the model from scratch, run the `train.py` script:

```bash
python train.py
```

You can customize the training process by modifying the hyperparameters such as epochs, learning rate, and batch size within the `train.py` script.

Making Predictions

To use the trained model to make a prediction on an image, use the `predict.py` script:

```bash
python predict.py --image /path/to/your/image.jpg
```

Model Architecture

The core of this project is a Convolutional Neural Network (CNN). The specific architecture of the network is defined in the Python scripts. The model is designed to take an image as input and output a probability score for the image being "real" or "fake".

Results

The model was trained for 21 epochs and achieved a **best validation accuracy of 65.15%**. The training was halted by the early stopping mechanism when the validation loss did not improve for 10 consecutive epochs.

Training Progress

Here are screenshots of the training process:

**Epochs 1-13:**
[](https://github.com/Jaiamar/AI-Image-Forgery-Detection/blob/main/__pycache__/Screenshot%202025-10-11%20050941.png)

**Epochs 17-21 and Final Results:**
[](https://github.com/Jaiamar/AI-Image-Forgery-Detection/blob/main/__pycache__/Screenshot%202025-10-11%20051748.png)

Contributing

Contributions are welcome\! If you have any suggestions, bug reports, or want to contribute to the code, please feel free to open an issue or submit a pull request.

License

This project is licensed under the MIT License. See the `LICENSE` file for more details. *(It's a good practice to add a `LICENSE` file to your repository.)*

