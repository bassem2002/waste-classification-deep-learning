# Waste Classification using Deep Learning

Deep Learning academic project focused on image-based waste classification using **TensorFlow/Keras**, **ResNet50**, transfer learning, fine-tuning, and custom CNN blocks inspired by DenseNet.

The project was developed as part of a Deep Learning practical assignment on **CNN architectures and fine-tuning**.

---

## Project Objective

The objective of this project is to build and adapt deep learning models for waste image classification.

The work focuses on:

- Using a pretrained **ResNet50** model as a feature extractor
- Applying transfer learning and fine-tuning
- Performing image data augmentation
- Building a binary waste classifier
- Designing custom DenseNet-inspired convolutional blocks
- Combining extracted features from different CNN stages
- Training and validating the model using TensorFlow datasets
- Applying Early Stopping and Model Checkpointing

---

## Dataset

The project uses a waste image dataset named `g_dechets`.

Images are loaded in RGB format with the following dimensions:

```text
48 × 48 × 3

The dataset is divided into:
- Training dataset
- Validation dataset
- Test dataset
20% of the training data is automatically reserved for validation.
The model performs a binary classification between two categories:
O
R

The original notebook does not explicitly describe what the labels O and R represent.
Data Preprocessing
The images are loaded using TensorFlow's:
tf.keras.preprocessing.image_dataset_from_directory()


Configuration:
Image size      : 48 × 48
Color mode      : RGB
Batch size      : 64
Validation split: 20%
Seed            : 123

The datasets are optimized using TensorFlow's data pipeline:
.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


This improves training performance by caching data in memory and preparing batches asynchronously.
Data Augmentation
A data augmentation pipeline is applied before feature extraction.
The transformations include:
- Random rotation
- Random zoom
- Horizontal flipping
Implementation:
data_augmentation = tf.keras.Sequential([    tf.keras.layers.RandomRotation(0.3),    tf.keras.layers.RandomZoom(0.5),    tf.keras.layers.RandomFlip(mode='horizontal')])


Data augmentation helps improve model generalization by creating variations of the training images.
ResNet50 Feature Extraction
The project uses ResNet50 pretrained on ImageNet without its original classification head.
tf.keras.applications.resnet.ResNet50(    include_top=False,    weights='imagenet',    input_shape=(48, 48, 3))


The pretrained convolutional layers are used to extract visual features from waste images.
Garbage Classification Model
A first model named cls_garbage is created using ResNet50.
Its architecture includes:
1. Data augmentation
2. ResNet50 convolutional feature extractor
3. Global Average Pooling
4. Dense layer with 128 neurons
5. L1/L2 regularization
6. Batch Normalization
7. ReLU activation
8. Additional Dense layer
9. Dropout
10. Batch Normalization
11. ReLU activation
12. Softmax classification layer
The original classifier predicts 6 classes.
The final layer is:
Dense(6, activation='softmax')


Pretrained weights are loaded from:
garbage_resnet.h5

Binary Classification Model
A second model named p_m_dechets1 is created from the pretrained garbage classification model.
The original feature extraction architecture is reused while replacing the final classification layer.
The new output layer contains:
Dense(1, activation='sigmoid')


This converts the original multiclass model into a binary classifier for categories O and R.
Custom Dense Block
The project also implements a custom DenseNet-inspired block.
The block contains:
Batch Normalization
↓
ReLU
↓
1 × 1 Convolution — 128 filters
↓
Batch Normalization
↓
ReLU
↓
3 × 3 Convolution — 32 filters
↓
Concatenation with the original feature map
↓
Batch Normalization
↓
ReLU

This approach allows the network to reuse previously extracted features.
Transition Layer
A custom transition block is also implemented.
It contains:
1 × 1 Convolution
↓
Batch Normalization
↓
ReLU
↓
Average Pooling 2 × 2

The 1 × 1 convolution reduces the number of channels by half.
Average Pooling reduces the spatial dimensions of the feature maps.
Final CNN Architecture
The final model is named:
p_m_dechets2

It combines pretrained ResNet50 features with custom DenseNet-inspired blocks.
The architecture includes:
- Data augmentation
- Partial ResNet50 feature extraction
- Frozen pretrained layers
- Intermediate ResNet50 feature extraction
- Five custom Dense Blocks
- Transition Layers
- Feature concatenation
- 1 × 1 convolution
- Residual feature addition
- Global Average Pooling
- Dropout
- Dense layer with 256 neurons
- Batch Normalization
- ReLU activation
- Binary classification layer using sigmoid
The final output layer is:
Dense(1, activation='sigmoid')


Transfer Learning
The project applies transfer learning by reusing features extracted from the pretrained ResNet50 network.
Some layers are frozen using:
extract1.trainable = False


This prevents pretrained weights from being modified during part of the training process.
Other layers remain available for adaptation to the new waste classification task.
Model Training
The model is compiled using:
optimizer = 'adam'loss = 'binary_crossentropy'metrics = ['accuracy']


The training process monitors both:
accuracy
val_accuracy

Early Stopping
Early Stopping is used to stop training if the validation accuracy does not improve for 3 consecutive epochs.
tf.keras.callbacks.EarlyStopping(    monitor='val_accuracy',    patience=3,    restore_best_weights=True)


This reduces unnecessary training and helps limit overfitting.
Model Checkpoint
The best model weights are automatically saved according to validation accuracy.
tf.keras.callbacks.ModelCheckpoint(    filepath='p_m_dechets2.weights.h5',    monitor='val_accuracy',    mode='max',    save_weights_only=True,    save_best_only=True)


Only the best-performing weights are retained.
Technologies
- Python
- TensorFlow
- Keras
- ResNet50
- Convolutional Neural Networks
- Transfer Learning
- Fine-Tuning
- DenseNet concepts
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Jupyter Notebook
- Google Colab
Project Structure
waste-classification-deep-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
└── waste_classification_resnet50.ipynb

Large datasets and trained model files are excluded from the Git repository.
Installation
Clone the repository:
git clone https://github.com/bassem2002/waste-classification-deep-learning.git

Move into the project folder:
cd waste-classification-deep-learning

Install the dependencies:
pip install -r requirements.txt

Running the Notebook
Start Jupyter Notebook:
jupyter notebook

Then open:
waste_classification_resnet50.ipynb

The project was originally developed in Google Colab and includes Google Drive mounting instructions for loading datasets and pretrained model weights.
Requirements
The main Python dependencies are:
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
jupyter

Notes
The notebook depends on external files that are not included in this repository:
garbage_resnet.h5
p_m_dechets1.h5
Train/
Test/

These files must be provided separately before all notebook cells can be executed successfully.
The original notebook also contains references to Google Drive paths, which may need to be adapted when running the project locally.
Academic Context
This project was developed as part of a Deep Learning practical assignment:
Deep Learning
Practical Work No. 4
CNN Architectures / Fine-Tuning
Academic Year 2025–2026

Author
Bassem Wali
Computer Engineering Student
Specialization: Software Engineering and Business Intelligence
GitHub:
https://github.com/bassem2002
