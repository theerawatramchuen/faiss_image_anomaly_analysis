import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ImageAnomalyDetector:
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_pretrained_model()
        self.transform = self._get_transform()
        self.index = None
        self.scaler = StandardScaler()
        self.train_features = []
        
    def _load_pretrained_model(self):
        """Load pre-trained ResNet model for feature extraction"""
        model = models.resnet18(pretrained=True)
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Define image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_path):
        """Extract features from an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
                
            return features.flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def train(self, train_folder):
        """Train the anomaly detector on normal images"""
        print("Extracting features from training images...")
        train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Extract features from all training images
        features_list = []
        valid_files = []
        
        for file_path in tqdm(train_files, desc="Training"):
            features = self.extract_features(file_path)
            if features is not None:
                features_list.append(features)
                valid_files.append(file_path)
        
        if not features_list:
            raise ValueError("No valid training images found!")
        
        self.train_features = np.array(features_list)
        
        # Normalize features
        self.train_features = self.scaler.fit_transform(self.train_features)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.train_features.shape[1])
        self.index.add(self.train_features.astype('float32'))
        
        print(f"Training completed. Index built with {len(self.train_features)} images.")
    
    def compute_anomaly_score(self, features):
        """Compute anomaly score based on distance to nearest neighbors"""
        if self.index is None:
            raise ValueError("Model not trained! Call train() first.")
        
        # Normalize features
        features = self.scaler.transform(features.reshape(1, -1)).astype('float32')
        
        # Search for k nearest neighbors
        k = min(5, len(self.train_features))
        distances, _ = self.index.search(features, k)
        
        # Anomaly score is the average distance to k nearest neighbors
        anomaly_score = np.mean(distances[0])
        return anomaly_score
    
    def create_anomaly_heatmap(self, image_path, patch_size=56):
        """Create anomaly heatmap by analyzing image patches"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_h, original_w = image_rgb.shape[:2]
            
            # Resize for processing
            processed_image = cv2.resize(image_rgb, (224, 224))
            h, w = processed_image.shape[:2]
            
            # Create heatmap
            heatmap = np.zeros((h, w))
            
            # Analyze patches
            for i in range(0, h - patch_size + 1, patch_size // 2):
                for j in range(0, w - patch_size + 1, patch_size // 2):
                    # Extract patch
                    patch = processed_image[i:i+patch_size, j:j+patch_size]
                    
                    # Convert to PIL Image and extract features
                    patch_pil = Image.fromarray(patch)
                    patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        patch_features = self.model(patch_tensor)
                        patch_features = patch_features.squeeze().cpu().numpy().flatten()
                    
                    # Compute anomaly score for patch
                    if self.index is not None:
                        patch_features_norm = self.scaler.transform(
                            patch_features.reshape(1, -1)).astype('float32')
                        distances, _ = self.index.search(patch_features_norm, 1)
                        patch_score = distances[0][0]
                        
                        # Update heatmap
                        heatmap[i:i+patch_size, j:j+patch_size] = np.maximum(
                            heatmap[i:i+patch_size, j:j+patch_size], patch_score)
            
            # Resize heatmap to original size
            heatmap = cv2.resize(heatmap, (original_w, original_h))
            
            # Normalize heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            return heatmap, image_rgb
            
        except Exception as e:
            print(f"Error creating heatmap for {image_path}: {e}")
            return None, None
    
    def analyze_images(self, input_folder, output_folder):
        """Analyze all images in input folder and save anomaly heatmaps"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images to analyze...")
        
        for filename in tqdm(image_files, desc="Analyzing images"):
            input_path = os.path.join(input_folder, filename)
            
            # Compute overall anomaly score
            features = self.extract_features(input_path)
            if features is None:
                continue
                
            anomaly_score = self.compute_anomaly_score(features)
            
            # Create heatmap
            heatmap, original_image = self.create_anomaly_heatmap(input_path)
            
            if heatmap is not None and original_image is not None:
                # Create output filename
                name, ext = os.path.splitext(filename)
                score_str = f"{anomaly_score:05.1f}"
                output_filename = f"{score_str}_{name}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                
                # Create and save heatmap visualization
                self._save_heatmap(original_image, heatmap, output_path, anomaly_score)
    
    def _save_heatmap(self, original_image, heatmap, output_path, anomaly_score):
        """Save heatmap visualization"""
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Anomaly Heatmap')
        plt.axis('off')
        plt.colorbar()
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(original_image)
        plt.imshow(heatmap, cmap='hot', alpha=0.5)
        plt.title(f'Overlay (Score: {anomaly_score:.2f})')
        plt.axis('off')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    # Initialize detector
    detector = ImageAnomalyDetector()
    
    # Train on normal images
    train_folder = "images_train"
    if not os.path.exists(train_folder):
        print(f"Training folder '{train_folder}' not found!")
        return
    
    detector.train(train_folder)
    
    # Analyze images
    input_folder = "images_All"
    output_folder = "anomaly_output"
    
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found!")
        return
    
    detector.analyze_images(input_folder, output_folder)
    print(f"Analysis complete! Results saved to '{output_folder}'")

if __name__ == "__main__":
    main()