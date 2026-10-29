# Lab Mid Viva Notes: HOG vs SIFT on Blood-Cell Images

## 1. Project overview

This notebook compares two image-feature extraction methods for classifying four BloodMNIST blood-cell classes:

- **HOG** — Histogram of Oriented Gradients
- **SIFT** — Scale-Invariant Feature Transform

Both methods are paired with a **Linear Support Vector Classifier (LinearSVC)**. The experiment compares validation performance and then tests robustness to image rotation and Gaussian noise.

The four selected classes are:

| Class ID | Class name |
|---:|---|
| 1 | Eosinophil |
| 5 | Monocyte |
| 6 | Neutrophil |
| 7 | Platelet |

The notebook uses a fixed random seed derived from registration number `059`.

## 2. Registration number and settings

- Registration number: `059`
- Random seed: `59`
- Last digit: `9`
- Since `9` is odd, HOG uses a cell size of **16 × 16 pixels**.
- SIFT uses a **6 × 6 fixed grid**, giving 36 grid keypoints.

A random seed helps make random sampling and noise generation reproducible.

## 3. Libraries used

- `medmnist`: loads the BloodMNIST dataset.
- `cv2` / OpenCV: image resizing, grayscale conversion, and SIFT.
- `scikit-image`: HOG and image transformations.
- `scikit-learn`: `LinearSVC`, `StandardScaler`, and evaluation metrics.
- `numpy`: numerical arrays and random sampling.
- `pandas`: tabular data.
- `matplotlib`: plotting and image display.

## 4. Dataset preparation

The notebook loads the official BloodMNIST training, validation, and test splits at size 64, then selects four classes.

| Split | Images per class | Total images |
|---|---:|---:|
| Training | 100 | 400 |
| Validation | 30 | 120 |
| Test | 30 | 120 |
| **Total** | **160** | **640** |

The selected images are resized to **128 × 128 pixels** and converted to grayscale.

### Why grayscale?

It simplifies processing because the feature-extraction pipelines use intensity information rather than separate RGB channels.

### Why resize to 128 × 128?

All images need a consistent size so that their feature vectors have consistent dimensions.

### Helper functions

- `unpack_dataset(images, labels)`: extracts image arrays and flattens labels into a one-dimensional integer array.
- `prepare_split(images, labels, per_class, seed)`: randomly selects a fixed number of images from each class without replacement, then shuffles them.
- `to_gray128(image)`: converts an image to grayscale, resizes it to 128 × 128 using cubic interpolation, and returns an 8-bit image.

## 5. HOG feature extraction

**HOG** stands for *Histogram of Oriented Gradients*. It describes local image shape using directions of intensity gradients.

### HOG settings

- Image size: 128 × 128
- Pixels per cell: 16 × 16
- Orientations: 9
- Cells per block: 2 × 2
- Block normalization: `L2-Hys`

### Why are there 1,764 HOG features?

1. There are 128 / 16 = 8 cells along each image dimension.
2. A 2 × 2 block can be placed in 7 positions horizontally and 7 vertically.
3. Total blocks: 7 × 7 = 49.
4. Each block contains 2 × 2 × 9 = 36 values.
5. Total features: 49 × 36 = **1,764**.

HOG captures edge directions and local shape patterns. It does not directly understand the meaning of a blood cell; it converts the image into numerical features.

### Why normalize HOG blocks?

Block normalization helps reduce the influence of overall contrast and illumination changes on gradient magnitudes.

## 6. SIFT feature extraction

**SIFT** stands for *Scale-Invariant Feature Transform*. It detects distinctive image locations and calculates descriptors around them.

### Normal SIFT results

The notebook's normal SIFT detector found:

- Average keypoints per image: **114.7975**
- Minimum: **12**
- Maximum: **321**
- Images with zero keypoints: **0**

Normal SIFT can produce a different number of descriptors for each image. A conventional SVM requires a fixed number of features for every sample, so variable-length descriptor sets cannot be passed directly as one standard feature matrix.

### Fixed-grid SIFT

The notebook places keypoints on a **6 × 6 grid**.

- Grid keypoints per image: 6 × 6 = 36
- Descriptor length per keypoint: 128 values
- Total feature length: 36 × 128 = **4,608**

Each image therefore has a **4,608-element SIFT feature vector**.

If OpenCV returns fewer descriptors than expected, the notebook uses zero-padding to keep the vector length fixed.

**Implementation note:** the notebook copies returned descriptors into the first rows of a zero-filled descriptor array. If OpenCV drops some grid keypoints, the remaining descriptors are not explicitly mapped back to their original grid positions.

## 7. Standardization and classifier

Both feature methods use the same pipeline:

1. `StandardScaler`: standardizes features so they are on comparable scales.
2. `LinearSVC`: learns a linear decision boundary to classify the images.

### What is an SVM?

A Support Vector Machine is a supervised machine-learning method that learns a boundary separating classes. `LinearSVC` uses a linear decision boundary.

### What is `C`?

`C` controls the trade-off between fitting the training data and regularization.

- Smaller `C`: stronger regularization.
- Larger `C`: weaker regularization and more emphasis on fitting training examples.

The notebook tests `C = 0.1`, `1`, and `10`. It selects based on validation macro-F1, using validation accuracy as a secondary sorting measure. All tested values tied in the recorded validation results, so `C = 0.1` was selected first by the sorting order.

### Why use a pipeline?

It keeps scaling and classification together and ensures the scaler is fitted on training data rather than independently on validation or test data.

## 8. Validation results

### HOG + LinearSVC

- Selected `C`: **0.1**
- Validation accuracy: **70.83%**
- Validation macro-F1: **0.70936**
- Correct predictions: **85 out of 120**

Confusion matrix (rows = true class, columns = predicted class; order: eosinophil, monocyte, neutrophil, platelet):

```text
[[17, 9, 4, 0],
 [ 7,21, 2, 0],
 [ 9, 4,17, 0],
 [ 0, 0, 0,30]]
```

The largest single-direction confusion was **true eosinophil predicted as monocyte**, occurring 9 times.

### SIFT + LinearSVC

- Selected `C`: **0.1**
- Validation accuracy: **77.50%**
- Validation macro-F1: **0.77519**
- Correct predictions: **93 out of 120**

Confusion matrix:

```text
[[18, 4, 7, 1],
 [ 8,22, 0, 0],
 [ 6, 0,23, 1],
 [ 0, 0, 0,30]]
```

The largest single-direction confusion was **true monocyte predicted as eosinophil**, occurring 8 times.

### What is a confusion matrix?

- Each row represents the actual class.
- Each column represents the predicted class.
- Values on the diagonal are correct predictions.
- Values outside the diagonal are errors.

### What is macro-F1?

F1 combines precision and recall. Macro-F1 calculates F1 separately for each class and then averages the class scores equally.

## 9. Robustness experiment

The notebook freezes the selected models and evaluates the test set under three conditions:

1. **Clean:** original test images.
2. **Rotated:** images rotated by 20 degrees.
3. **Noisy:** Gaussian noise added with σ = 0.05 after scaling pixel values to the range 0–1.

The models are not retrained during these robustness tests.

### Test accuracy

| Method | Clean | Rotated 20° | Gaussian noise |
|---|---:|---:|---:|
| HOG + LinearSVC | 70.83% | 70.00% | 25.83% |
| SIFT + LinearSVC | 75.00% | 71.67% | 75.00% |

### Interpretation

- On clean test images, SIFT scored 75.00%, compared with HOG at 70.83%.
- Rotation reduced HOG accuracy slightly, from 70.83% to 70.00%.
- Rotation reduced SIFT accuracy from 75.00% to 71.67%.
- Gaussian noise caused HOG accuracy to fall sharply to 25.83%.
- SIFT remained at 75.00% in the recorded noisy-test result.

For this particular experiment, SIFT performed better overall, especially under the specified Gaussian-noise condition. This does **not** prove SIFT is always superior to HOG; results depend on the dataset, preprocessing, settings, and implementation.

## 10. Short-answer viva questions

### Q1. What is the main aim of the experiment?

To compare HOG and fixed-grid SIFT features with a LinearSVC classifier on four BloodMNIST classes, including robustness tests using rotation and Gaussian noise.

### Q2. Why convert images to grayscale?

It simplifies the input and provides intensity information for feature extraction.

### Q3. Why resize all images to 128 × 128?

Consistent dimensions help keep feature extraction and vector sizes consistent.

### Q4. Why does HOG produce 1,764 features?

There are 49 overlapping 2 × 2-cell blocks, and each block has 36 orientation values: 49 × 36 = 1,764.

### Q5. Why does fixed-grid SIFT produce 4,608 features?

There are 36 grid keypoints, each with a 128-value descriptor: 36 × 128 = 4,608.

### Q6. Why not feed normal SIFT descriptors directly into the SVM?

The number of detected keypoints varies by image, so the number of descriptors is not fixed. A standard feature matrix needs the same feature length for every sample.

### Q7. Why use a fixed grid for SIFT?

It aims to produce a consistent number of keypoints and a fixed-length descriptor vector for every image.

### Q8. What does `StandardScaler` do?

It standardizes each feature using statistics learned from the training set.

### Q9. What is the role of `LinearSVC`?

It learns a linear classification boundary from the extracted feature vectors.

### Q10. What is the role of `C`?

It controls the regularization-versus-training-fit trade-off in the SVM.

### Q11. Why select `C` using validation data?

Validation data helps choose hyperparameters without using the test set. Tuning on the test set can bias the final evaluation.

### Q12. Why use macro-F1?

It gives equal weight to each class's F1 score.

### Q13. What does a confusion matrix show?

It shows how many samples from each true class were predicted as each class.

### Q14. What is the difference between validation and test data?

Validation data is used for model selection. Test data is used for final evaluation after the model and settings are frozen.

### Q15. What is Gaussian noise?

Random noise sampled from a Gaussian (normal) distribution and added to pixel intensities.

### Q16. What does σ = 0.05 mean here?

The noise standard deviation is 0.05 when pixel intensities are represented on a 0–1 scale.

### Q17. Which method performed better on the clean test set?

SIFT: 75.00%, compared with HOG at 70.83%.

### Q18. Which method was more robust to the specified Gaussian noise?

SIFT in this experiment: its accuracy stayed at 75.00%, while HOG fell to 25.83%.

### Q19. Did rotation affect both methods?

Yes. HOG changed from 70.83% to 70.00%, and SIFT changed from 75.00% to 71.67%.

### Q20. Can we conclude SIFT is always better than HOG?

No. We can only conclude that SIFT performed better under the conditions tested in this notebook.

## 11. Common viva definitions

- **Feature extraction:** converting an image into numerical values that describe useful patterns.
- **Classifier:** a model that assigns an input to a class.
- **Training set:** data used to fit the model.
- **Validation set:** data used to select model settings.
- **Test set:** data reserved for final evaluation.
- **Accuracy:** correct predictions divided by total predictions.
- **Precision:** among items predicted as a class, the fraction that truly belongs to that class.
- **Recall:** among items truly belonging to a class, the fraction correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Robustness:** how well a model performs when inputs are altered or corrupted.
- **Keypoint:** a distinctive location detected in an image.
- **Descriptor:** numerical representation of the image region around a keypoint.
- **Random seed:** a fixed starting value that makes pseudo-random operations reproducible.

## 12. Limitations and points to remember

1. The experiment uses four selected classes, not every BloodMNIST class.
2. The dataset is sampled to 400 training, 120 validation, and 120 test images.
3. Results describe this specific subset and configuration; they may differ with other samples or settings.
4. Normal SIFT detection and fixed-grid SIFT descriptor construction are different steps in the notebook.
5. The fixed-grid implementation uses zero-padding if fewer descriptors are returned, but does not explicitly preserve the original grid slot for every returned descriptor.
6. All three `C` values tied in the recorded validation results, so `C = 0.1` was selected by sorting order rather than a uniquely higher score.
7. The noise result is specific to the stated Gaussian-noise strength and random seed.
8. Do not tune parameters using test-set performance.

## 13. One-minute viva summary

“My project compares HOG and fixed-grid SIFT feature extraction for four BloodMNIST blood-cell classes. I selected 400 training images, 120 validation images, and 120 test images, with equal numbers from each class. Every image was converted to grayscale and resized to 128 by 128 pixels. HOG generated 1,764 features per image, while fixed-grid SIFT generated 4,608 features. I standardized the features and trained a LinearSVC, selecting the regularization parameter using validation macro-F1. SIFT achieved 77.50% validation accuracy, compared with HOG at 70.83%. On the test set, SIFT achieved 75.00% accuracy on clean and noisy images, while HOG achieved 70.83% on clean images and 25.83% under Gaussian noise. These results show that SIFT performed better in this experiment, but they do not establish that it is universally better than HOG.”

## 14. Final revision checklist

- [ ] Memorize the meanings of HOG, SIFT, SVM, and `C`.
- [ ] Memorize feature lengths: HOG = 1,764; SIFT = 4,608.
- [ ] Memorize data split: 400 train, 120 validation, 120 test.
- [ ] Understand the difference between validation and test sets.
- [ ] Be able to read a confusion matrix.
- [ ] Remember clean, rotated, and noisy test accuracies.
- [ ] Explain why fixed-grid SIFT gives a fixed-length representation.
- [ ] State conclusions only for this experiment, not as universal claims.
