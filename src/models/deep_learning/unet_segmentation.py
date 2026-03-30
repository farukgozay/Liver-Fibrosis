"""
U-Net Segmentation Model for Liver and Spleen
==============================================

Professional deep learning-based segmentation using U-Net architecture.
Automatically segments liver and spleen from abdominal CT images.

Author: Bülent Tuğrul
Institution: Ankara Üniversitesi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import cv2


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Medical Image Segmentation
    
    Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Ronneberger et al., 2015
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 3):
        """
        Initialize U-Net
        
        Parameters:
        -----------
        n_channels : int
            Number of input channels (1 for grayscale CT)
        n_classes : int
            Number of output classes (3: background, liver, spleen)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder (Expansive Path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        # Output
        logits = self.outc(x)
        
        return logits


class LiverSpleenSegmentor:
    """
    Liver and Spleen Segmentation System
    
    Uses trained U-Net model for automatic organ segmentation
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Segmentor
        
        Parameters:
        -----------
        model_path : str, optional
            Path to pre-trained model weights
        device : str
            'cuda' or 'cpu'
        """
        self.device = device
        self.model = UNet(n_channels=1, n_classes=3).to(device)
        
        if model_path:
            self.load_model(model_path)
        else:
            print("⚠️ No pre-trained model loaded. Using random weights.")
            print("   Train model or provide model_path for production use.")
        
        self.model.eval()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for U-Net
        
        Parameters:
        -----------
        image : np.ndarray
            Input CT image (H, W)
            
        Returns:
        --------
        tensor : torch.Tensor
            Preprocessed tensor (1, 1, H, W)
        """
        # Normalize to [0, 1]
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Resize to standard size (512x512)
        image_resized = cv2.resize(image_norm, (512, 512))
        
        # Convert to tensor
        tensor = torch.from_numpy(image_resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return tensor.to(self.device)
    
    def postprocess(self, 
                    output: torch.Tensor,
                    original_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess U-Net output
        
        Parameters:
        -----------
        output : torch.Tensor
            Model output (1, 3, H, W)
        original_size : tuple
            Original image size to resize back to
            
        Returns:
        --------
        liver_mask : np.ndarray
            Binary liver mask
        spleen_mask : np.ndarray
            Binary spleen mask
        """
        # Get predictions
        probs = F.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        
        # Extract liver and spleen
        liver_mask = (preds == 1).astype(np.uint8)
        spleen_mask = (preds == 2).astype(np.uint8)
        
        # Resize to original size
        liver_mask = cv2.resize(liver_mask, (original_size[1], original_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
        spleen_mask = cv2.resize(spleen_mask, (original_size[1], original_size[0]),
                                interpolation=cv2.INTER_NEAREST)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_CLOSE, kernel)
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_OPEN, kernel)
        
        spleen_mask = cv2.morphologyEx(spleen_mask, cv2.MORPH_CLOSE, kernel)
        spleen_mask = cv2.morphologyEx(spleen_mask, cv2.MORPH_OPEN, kernel)
        
        return liver_mask, spleen_mask
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment liver and spleen from CT image
        
        Parameters:
        -----------
        image : np.ndarray
            Input CT image
            
        Returns:
        --------
        liver_mask : np.ndarray
            Liver segmentation mask
        spleen_mask : np.ndarray
            Spleen segmentation mask
        """
        original_size = image.shape
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        liver_mask, spleen_mask = self.postprocess(output, original_size)
        
        return liver_mask, spleen_mask
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"✓ Loaded model from {model_path}")
    
    def save_model(self, model_path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Saved model to {model_path}")


# Fallback: Traditional Image Processing Segmentation
class TraditionalSegmentor:
    """
    Fallback segmentation using traditional image processing
    
    Used when U-Net model is not available
    """
    
    @staticmethod
    def segment_liver_traditional(image: np.ndarray) -> np.ndarray:
        """
        Segment liver using thresholding and morphology
        
        Parameters:
        -----------
        image : np.ndarray
            CT image in HU
            
        Returns:
        --------
        liver_mask : np.ndarray
            Binary liver mask
        """
        # Liver typically 40-70 HU
        liver_mask = ((image > 30) & (image < 100)).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(liver_mask, connectivity=8)
        if num_labels > 1:
            # Find largest (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            liver_mask = (labels == largest_label).astype(np.uint8)
        
        # Convex hull to fill holes
        contours, _ = cv2.findContours(liver_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            liver_mask_filled = np.zeros_like(liver_mask)
            cv2.drawContours(liver_mask_filled, [hull], -1, 1, -1)
            liver_mask = liver_mask_filled
        
        return liver_mask
    
    @staticmethod
    def segment_spleen_traditional(image: np.ndarray, 
                                   liver_mask: np.ndarray) -> np.ndarray:
        """
        Segment spleen using anatomical constraints
        
        Parameters:
        -----------
        image : np.ndarray
            CT image in HU
        liver_mask : np.ndarray
            Liver mask for anatomical reference
            
        Returns:
        --------
        spleen_mask : np.ndarray
            Binary spleen mask
        """
        # Spleen typically 45-55 HU, left side of image
        spleen_mask = ((image > 40) & (image < 70)).astype(np.uint8)
        
        # Remove liver region
        spleen_mask = spleen_mask * (1 - liver_mask)
        
        # Focus on left side (spleen is typically on left)
        h, w = spleen_mask.shape
        spleen_mask[:, w//2:] = 0  # Zero out right half
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        spleen_mask = cv2.morphologyEx(spleen_mask, cv2.MORPH_CLOSE, kernel)
        spleen_mask = cv2.morphologyEx(spleen_mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest component in left region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spleen_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            spleen_mask = (labels == largest_label).astype(np.uint8)
        
        return spleen_mask


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("U-Net Liver & Spleen Segmentation Module")
    print("="*80)
    
    # Test with random data
    test_image = np.random.randn(512, 512) * 30 + 50
    
    print("\nOption 1: U-Net Deep Learning Segmentation")
    print("-" * 80)
    try:
        segmentor = LiverSpleenSegmentor()
        liver, spleen = segmentor.segment(test_image)
        print(f"✓ U-Net segmentation successful")
        print(f"  Liver pixels: {np.sum(liver)}")
        print(f"  Spleen pixels: {np.sum(spleen)}")
    except Exception as e:
        print(f"⚠️ U-Net failed: {e}")
    
    print("\nOption 2: Traditional Image Processing Segmentation")
    print("-" * 80)
    traditional = TraditionalSegmentor()
    liver_trad = traditional.segment_liver_traditional(test_image)
    spleen_trad = traditional.segment_spleen_traditional(test_image, liver_trad)
    print(f"✓ Traditional segmentation successful")
    print(f"  Liver pixels: {np.sum(liver_trad)}")
    print(f"  Spleen pixels: {np.sum(spleen_trad)}")
    
    print("\n" + "="*80)
    print("✅ Segmentation Module Ready!")
    print("="*80)
