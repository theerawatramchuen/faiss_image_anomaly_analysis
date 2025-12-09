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
Optional for speed performance with machine Cuda GPU
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
Processes all images in "images_All" folder <br>
Computes anomaly scores based on distance to normal feature clusters <br>
Generates heatmaps by analyzing image patches <br>
#### Output:
Saves heatmap visualizations to "anomaly_output" folder <br>
Filename format: {anomaly_score}_{original_name}.jpg <br>
Includes original image, heatmap, and overlay in output <br>
#### Progress Tracking:
Uses tqdm for progress bars during training and inference <br>
#### Usage:
Place normal training images in images_train/ folder <br>
Place test images in images_All/ folder <br>
Run the script <br>
Check anomaly_output/ for results <br><br>
The anomaly score represents how different an image is from the normal training set, with higher scores indicating more anomalous content. The heatmap shows which regions of the image contribute most to the anomaly score.

### Future Development Recommendations
### 1. Enhanced Feature Extraction
```
python
# Consider using more powerful backbones
model = models.resnet50(pretrained=True)
# Or vision transformers
model = models.vit_b_16(pretrained=True)
```
### 2. Batch Processing
```
# Implement batch feature extraction for speed improvement
def extract_features_batch(self, image_paths):
    """Extract features from multiple images in batch"""
    batch_tensors = []
    # ... batch processing logic
```
### 3. Advanced Anomaly Detection Methods
* Implement autoencoder-based reconstruction error
* Add support for One-Class SVM
* Incorporate Mahalanobis distance for anomaly scoring
### 4. Configuration Management
```
python
# Add config class for easy parameter tuning
class Config:
    patch_size = 56
    k_neighbors = 5
    similarity_metric = 'l2'  # or 'cosine'
    backbone = 'resnet18'
```
### 5. Model Persistence
```
python
def save_model(self, path):
    """Save trained model and index"""
    torch.save({
        'scaler': self.scaler,
        'train_features': self.train_features,
        'config': self.config
    }, path)
    
    faiss.write_index(self.index, path + '.index')

def load_model(self, path):
    """Load pre-trained model"""
    # Implementation
```
### 6. Real-time Processing
```
python
def process_stream(self, video_source=0):
    """Process video stream in real-time"""
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        # Process frame and display anomaly score
```
### 7. Performance Metrics
```
python
def evaluate(self, normal_folder, anomaly_folder):
    """Evaluate detector performance"""
    # Calculate precision, recall, F1-score
    # Generate ROC curve
    # Compute AUC
```
### Example Usage Script
```
python
#!/usr/bin/env python3
"""
Example usage script with command-line arguments
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Image Anomaly Detection')
    parser.add_argument('--train', required=True, help='Training folder')
    parser.add_argument('--test', required=True, help='Test folder')
    parser.add_argument('--output', default='anomaly_output', help='Output folder')
    parser.add_argument('--patch-size', type=int, default=56, help='Patch size for heatmap')
    
    args = parser.parse_args()
    
    detector = ImageAnomalyDetector()
    detector.train(args.train)
    detector.analyze_images(args.test, args.output)

if __name__ == '__main__':
    main()
```



