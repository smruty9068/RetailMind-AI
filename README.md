# Day 7 – Transfer Learning & Object Detection
## RetailMind AI – Computer Vision Module

---

## 1. Overview

Day 7 extends the Computer Vision capabilities of RetailMind AI by exploring Transfer Learning and Object Detection.

The objective is to improve product image classification using pretrained CNN architectures such as VGG16 and ResNet50 and understand how object detection models such as YOLO can locate products within catalog images.

This increment builds upon the Fashion-MNIST image classification work introduced in Day 6.

## 2. Objectives

- Understand Transfer Learning and pretrained CNN architectures.
- Study VGG16 and ResNet50.
- Understand frozen layers and fine-tuning.
- Explore object localization using bounding boxes.
- Understand YOLO-based product detection.
- Identify real-world applications of object detection in retail.

## 3. Dataset

**Dataset:** Fashion-MNIST

Fashion-MNIST contains 70,000 grayscale clothing images belonging to 10 categories.

| Dataset | Images |
|---|---:|
| Training | 60,000 |
| Testing | 10,000 |
| Total | 70,000 |

Image dimensions: 28 × 28 pixels.

Categories:
1. T-shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

## 4. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Jupyter Notebook
- Fashion-MNIST

## 5. Transfer Learning

Transfer Learning is a technique in which a model trained on a large dataset is reused for a different but related task.

Instead of training a CNN from scratch, we reuse the knowledge learned by an existing model.

### Pretrained Architectures Studied

**VGG16**
- Uses multiple 3 × 3 convolution filters.
- Learns hierarchical image features.
- Commonly used for image classification.

**ResNet50**
- Uses residual connections.
- Helps train deeper neural networks.
- Reuses pretrained visual features.

### Transfer Learning Workflow

1. Load a pretrained CNN backbone.
2. Freeze the base convolutional layers.
3. Extract image features.
4. Add a new classification head.
5. Train the classification head.
6. Unfreeze selected top layers.
7. Fine-tune using a smaller learning rate.

### Proposed Architecture

Input Image (224 × 224 × 3)

↓

Pretrained ResNet50 Backbone

↓

Global Average Pooling

↓

Dense Layer (128 neurons, ReLU)

↓

Dropout (0.3)

↓

Output Layer (10 neurons, Softmax)

**Note:** This architecture was studied and documented. Actual pretrained-model training and fine-tuning were not executed in the current environment.

## 6. Object Detection

Object Detection identifies and locates objects within an image.

Unlike image classification, which assigns one category to an entire image, object detection identifies individual objects and their locations using bounding boxes.

### YOLO – You Only Look Once

YOLO is a real-time object detection framework.

Its main capabilities include:

- Object identification.
- Bounding-box prediction.
- Class prediction.
- Confidence scoring.
- Detection of multiple objects in one image.

### Retail Applications

1. **Catalog Quality Assurance**
   - Identify incorrectly framed product images.
   - Detect unexpected objects.

2. **Multi-Product Images**
   - Locate multiple products within one image.
   - Extract individual product regions.

3. **Shelf Analytics**
   - Locate products on retail shelves.
   - Support inventory monitoring.

4. **Visual Product Search**
   - Identify products within larger images.
   - Send detected regions to a classification module.

**Note:** A pretrained YOLO model was not executed because Ultralytics was unavailable in the current environment.

## 7. Product Localization Demonstration

A basic foreground-based localization technique was implemented using Pillow and NumPy.

### Implementation Steps

1. Load a Fashion-MNIST image.
2. Resize the image for visualization.
3. Convert the image into a NumPy array.
4. Apply a pixel-intensity threshold.
5. Identify foreground pixel coordinates.
6. Calculate minimum and maximum coordinates.
7. Draw a bounding box around the foreground object.

The resulting image demonstrates basic product localization.

**Important:** This is threshold-based localization, not trained YOLO object detection. It is suitable only for simple images with sufficiently contrasting backgrounds.

## 8. Project Structure

```text
RetailMind-AI/
│
├── Notebooks/
│   ├── Day01.ipynb
│   ├── Day02.ipynb
│   ├── Day03.ipynb
│   ├── Day04_Feature_Engineering.ipynb
│   ├── Day05_ANN_Neural_Networks.ipynb
│   ├── Day06.ipynb
│   └── Day07.ipynb
│
├── Models/
│   ├── day6_fashion_classifier.joblib
│   └── day7_product_localization.png
│
├── Src/
│   └── ...
│
└── README.md
```

## 9. Environment Limitations

The implementation was performed using browser-based Jupyter running Python 3.13.1 on WebAssembly (Emscripten).

The following libraries were unavailable:

- TensorFlow
- Keras
- PyTorch
- Torchvision
- Ultralytics YOLO

Therefore:

- Pretrained VGG16/ResNet50 training was not performed.
- Fine-tuning was not performed.
- YOLO inference was not performed.
- Product localization was demonstrated using threshold-based image processing.

These limitations are documented to distinguish completed implementation work from the planned deep-learning components.

## 10. Day 7 Outcomes

The following activities were completed:

- Studied Transfer Learning concepts.
- Explored VGG16 and ResNet50 architectures.
- Designed a proposed transfer-learning architecture.
- Studied YOLO and object detection concepts.
- Implemented basic product localization.
- Visualized bounding-box coordinates.
- Identified retail applications of object detection.

## 11. Future Improvements

- Execute ResNet50 or VGG16 transfer learning in a compatible environment.
- Fine-tune pretrained convolutional layers.
- Compare transfer-learning accuracy with the Day 6 baseline.
- Integrate pretrained YOLO for product detection.
- Implement multi-product detection.
- Integrate classification and object detection into one vision module.
- Explore product segmentation and Grad-CAM explainability.

---

**RetailMind AI – Building an End-to-End Retail Intelligence Platform**

Day 7: Transfer Learning & Object Detection
