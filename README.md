# Gastric Cancer Histopathology Tissue Imaging Machine Learning Project

We explore Convolutional Neural Networks (CNNs) for classifying gastric cancer tissue images into 8 tissue classes. Using the Gastric Cancer Histopathology Tissue Imaging dataset from Kaggle (link), we experiment with ResNet18, ResNet50, custom CNNs, and ConvNeXt, aiming to identify the model with the best classification performance.

We encounter challenges such as overfitting and low F1 scores. To address these, we experiment with:

- Adding or removing layers
- Dropout layers
- L2 regularization
- Freezing/unfreezing backbone layers in pre-trained CNNs

Despite these efforts, overfitting remains a challenge on this dataset.

## Code Structure:
```
├── src/
│   ├── explore_dataset.py     # Dataset visualization and EDA
│   ├── train.py               # Training script
│   ├── model.py               # CNN architectures
│   ├── model_utils.py         # Metrics, evaluation functions
│   └── preprocessing.py       # Dataset loading, splitting, transforms
├── model/                     # Final Model outputs
├── notebooks/                 # Jupyter notebooks used for exploration
├── data/                      # Instructions on how to initialize dataset
├── trial_weights/             # Weights saved during trials on Jupyter notebook
├── README.md
└── requirements.txt
```
Reproducibility: The repository includes requirements.txt. Instructions in this README allow full end-to-end reproduction.

## Dataset:
Source: Kaggle Gastric Cancer Histopathology Tissue Imaging Dataset (https://www.kaggle.com/datasets/orvile/gastric-cancer-histopathology-tissue-image-dataset/data)

Total Images: 31,000 histological images from 300 slides

Classes:
- ADI (Adipose)
- BACK (Non-tissue)
- DEB (Cellular waste)
- LYM (Lymphocytes)
- MUC (Mucus)
- MUS (Muscle)
- NORM (Normal Colon Mucosa / Healthy Tissue)
- STR (Cancer Associated Stroma)
- TUM (Tumor)

Data Split:
- Training: 70%
- Validation: 15%
- Test: 15%

The dataset is not uploaded due to size. Instructions for downloading and preparing it are provided in the data/ folder.


## Exploratory Data Analysis (EDA)
Class distibution was visualized to ensure balance and sample images per class were shown using explore_dataset.py. There were no missing files or corrupted files detected. 
<img width="1000" height="500" alt="Class Distribution" src="https://github.com/user-attachments/assets/6e379c83-32b8-4e7c-8d81-e6ca988ac38f" />
<img width="1280" height="692" alt="Samples" src="https://github.com/user-attachments/assets/c2e68b86-fc33-4c39-ba31-683c8572a683" />


## Model Development
### Architecture Selection
- Custom CNN: Baseline simple CNN with fully connected layers, led to very low accuracies and high loss. 
- ResNet18/50: Baseline common CNN model for comparison, capped out around 60% F1 Score without overfitting. ResNet50 did slightly better.
- ResNet18 + Custom Layers: Provided higher train F1 score but severaly overfitted the model. 
- ConvNeXt Tiny (Layers Frozen): Provided a F1 score around 60s as well
- ConvNeXt Tiny (4th layer unfrozen): Provided a overfitted loss and accuracy, F1 was around 69
- ConvNeXt Tiny (4th layer unfrozen + L2): Provided a overfitted loss and accuracy but F1 went up to 69
- Final Model: ConvNeXt Tiny (4th layer unfrozen + L2 + Rotation): Provided a lower...

### Pipeline
1. Data Loading + Processing
- Dataset is loaded from Kaggle Gastric Cancer Histopathology Dataset
- Transformations: Resized to 224, color jitter, random rotation, and normalization added (avoiding to flip)
- Split: 70 / 15 / 15
- Data Loaders: Batched and Shuffled for Training; Batched only for Validation/Test
2. Backbone Selection
- Utilized ConvNeXt Tiny to leverage pretrained representations
- Frozen Layers: Freeze lower layers to preserve pretrained features; and unfroze higher layers for fine-tuning
- Regularization: Dropout, L2 Weight Decay, and Data augmentation
- Optimization: Adam optimizer with learning rate of 1e-4
- Loss Function: Cross-Entropy for Multi-class Classification
- Mixed Precision: Used to speed up training and reduce GPU memory consumption
3. Evaluation
- Metrics: Train/Validation loss and Accuracy per Epoch; F1 Score, Precision, Recall, Confusion Matrix, and ROC-AUC for final evaluation

### Model Choices
We experimented with several convolutional neural network (CNN) architectures to classify gastric cancer histopathology images into 8 tissue classes.
#### Baseline CNN
- A simple CNN with 3 convolutional layers, ReLU activations, max pooling, and fully connected layers.
- Performance: Train/Validation Loss: 1.15 / 1.32, Accuracy: 0.57 / 0.49, F1 Score: 0.4582
- Reason: Easy to implement and interpret; serves as a baseline to compare more complex models.

#### ResNet18 / ResNet50
- Residual Networks (ResNets) use residual connections H(x)=F(x)+x to allow gradients to flow more easily during training, mitigating the vanishing gradient problem in deep networks.
- Common issue in CNNs is that because there are so many layers, the signal or the improvement from back propogation gets worse leading to the vanishing gradient. ResNet, instead of forcing each layer to learn full transformations, it only learns the difference between the input and the desired output. H(x) is the output, F(x) is what the layer learns and x is the shortcut basically it doesn't need to change much to get through. In medical imaging, this is quite useful because there are often subtle features that deeper networks would be better at, but we don't want the vanishing gradient.
    
##### **Architecture:** 
- Stacked residual blocks with convolution, batch normalization, ReLU, pooling, and global average pooling followed by a fully connected layer.
##### **Performance:** 
- Improved over baseline; adding custom layers increased overfitting (train accuracy ~98%, validation F1 ~0.69).

**Reason:** Widely used CNNs that enable deeper models without degradation; good for benchmarking.

#### ConvNeXt Tiny
- Modern CNN inspired by Vision Transformers, combining CNN efficiency with transformer-like design choices. ConvNeXt utilizes key components of transformers such as layer normalization instead of batch normalization, large patch sizes instead of smaller kernels and simple feed-forward blocks instead of CNN bottle necks, making it scalable. Furthermore instead of bottlenecking, it expands first and then compresses. All of this benefits medical imaging because of the expanding nature, leading to more power and detection of subtle differences and abnormalities. Furthermore, ConvNeXt that is pretrained on sets like ImageNet allow it for fine tuned tasks in medical imaging. 

##### Architecture:
- Stem: 4×4 convolution (stride 4) to reduce resolution
- 4 stages of repeated ConvNeXt blocks with halving feature map size
- Global average pooling → Classifier (LayerNorm + Linear)
- ConvNeXt Block: Depthwise 7×7 conv → LayerNorm → 1×1 conv (expand) → GELU → 1×1 conv (project back) → Residual addition

**Performance:** Similar F1 score as ResNet18/50, but less overfitting

**Reason:** Efficient for large-scale image datasets, modernized design reduces overfitting, and utilizes depthwise large kernels and LayerNorm for better generalization

Summary:
- Baseline CNN is interpretable but underperforms on complex histopathology images
- ResNet improves training stability and F1 score but can overfit with additional layers
- ConvNeXt Tiny balances accuracy and overfitting, leveraging modern CNN design inspired by transformers

### Training Setup
The parameters we used were as follows:
- Loss: Cross-Entropy
- Optimizer: Adam with weight decay (L2 Regularization)
- Epochs: 10
- Learning Rate: 1e-4
- Metrics Tracked: Train/Validation Loss + Accuracy, F1 Score, Precision, Recall, Confusion Matrix, ROC AUC
- Mixed Precision training used for efficiency

## Results
Metric	            Score
F1 Score (Macro)	0.6987
Precision (Macro)	0.7010
Recall (Macro)	    0.6980
ROC-AUC (Macro)	    0.9514
ROC-AUC (Weighted)	0.9514

The F1 Score of ~0.7 mean is pretty decent and on average is correctly identifying positve cases while trying to balance false positives and false negatives. There is still much improvement that needs to be made. The precision of 0.701 displays that around 70% of the preditions are correct, and there isn't any over-predicting happening in certain classes. A recall of 0.698 displays that the model does detect around 70% of true samples, but still there are many false negatives that get through. But since recall and precision are pretty close, they are balanced errors and does not bias false positives or false negatives. 

The ROC-AUC measuring the ability to rank postivie samples higher than negative ones is pretty good. It is very confident in distinguishing classes from the other classes with an ROC-AUC of 0.95. 

<img width="1000" height="600" alt="Loss" src="https://github.com/user-attachments/assets/c997bccd-ed43-4a70-a974-c185f730f576" />
<img width="1000" height="600" alt="Accuracy" src="https://github.com/user-attachments/assets/95ce32c3-c874-41a9-aecb-413a49e14f18" />

<img width="700" height="500" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/e35cb2d5-ad1b-491e-8f34-5c591e1911b0" />

This model that we landed on still had overfitting after attempting various techniques, but it was not as bad as some of the other models we have tested and still provided one of the highest F1 scores. We can also see that there were a few like ADI and LYM which had pretty good amount of true positives and that trend was shown through the other models tested as well. There were a few classes that the models did better at classifying.

## Instructions to Run
1. Clone Repository and navigate to src/
2. Install Requirements: 
```pip install -r requirements.txt```
3. Prepare the dataset directory as described above. 
4. Run EDA:
```python explore_dataset.py```
5. Train models:
```python train.py```
6. Evaluate products from training and evaluation

## Technical Notes / Lessons Learned
Overfitting remains a major challenge due to the small number of slides relative to the dataset size. This leads to common tropes being learned in a slide and then over fitting on it. ConvNeXt Tiny performed slightly better than ResNet18/50 and performed just as well after adding various ways to reduce overfitting. While the final iteration provided a F1 score of around ___ which is about the same as ResNet18/50, when ran without certain features, ResNet18/50 overfitted more than ConvNeXt Tiny did. Experimenting with dropout and freezing/unfreezing layers helped reduce the overall overfitting issue, but further data augmentation or regularization may be needed. 






