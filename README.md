# FAISS Image Anomaly Analysis
### Installation
Python 3.12 
```
torch>=1.9.0
torchvision>=0.10.0
faiss-cpu>=1.7.0
opencv-python>=4.5.0
Pillow>=8.0.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.50.0
numpy>=1.19.0
```
Anomaly score 114.4
<img width="1766" height="598" alt="image" src="https://github.com/user-attachments/assets/2141b4f8-aa04-40f9-bdb8-749e667c558e" />
Anomaly score 1364.1
<img width="1766" height="598" alt="image" src="https://github.com/user-attachments/assets/9bb9ea7a-4c27-4f3e-bc74-de80798773c4" />

### Key Features:
#### Training Phase:
Extracts features from normal images in "images_train" using ResNet18 <br>
Builds a FAISS index for efficient similarity search <br>
#### Inference Phase:
Processes all images in "images_All" folder
Computes anomaly scores based on distance to normal feature clusters
Generates heatmaps by analyzing image patches
#### Output:
Saves heatmap visualizations to "anomaly_output" folder
Filename format: {anomaly_score}_{original_name}.jpg
Includes original image, heatmap, and overlay in output
#### Progress Tracking:
Uses tqdm for progress bars during training and inference
#### Usage:
Place normal training images in images_train/ folder
Place test images in images_All/ folder
Run the script
Check anomaly_output/ for results <br>
The anomaly score represents how different an image is from the normal training set, with higher scores indicating more anomalous content. The heatmap shows which regions of the image contribute most to the anomaly score.
