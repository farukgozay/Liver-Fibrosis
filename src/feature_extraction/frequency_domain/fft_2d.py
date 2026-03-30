"""
2D FFT (Fast Fourier Transform) Feature Extraction Module
==========================================================

This module implements 2D FFT-based frequency domain analysis for liver fibrosis
characterization. The frequency domain reveals texture periodicity and structural
patterns invisible in spatial domain.

Key Concepts:
-------------
- Spatial Domain: Traditional pixel-based image representation
- Frequency Domain: Fourier-transformed representation showing periodic patterns
- Low Frequencies: Overall structure and large-scale variations
- High Frequencies: Fine details, edges, and texture patterns

For NASH and Fibrosis Detection:
---------------------------------
- Fibr

otic tissue shows characteristic frequency signatures
- NASH-related steatosis alters frequency spectrum
- Cirrhotic patterns exhibit distinct high-frequency components

Author: Bülent Tuğrul
Institution: Ankara Üniversitesi - Bilgisayar Mühendisliği
Project: Karaciğer Fibrozunun BT Görüntülerinden Non-İnvaziv Evrelendirilmesi
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union
from scipy import ndimage, fftpack
from skimage import exposure
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class FFTFeatures:
    """Dataclass to store extracted FFT features"""
    
    # Power spectrum features
    total_power: float
    low_freq_power: float  # 0-25% of spectrum
    mid_freq_power: float  # 25-75% of spectrum
    high_freq_power: float  # 75-100% of spectrum
    
    # Frequency ratios (critical for fibrosis)
    low_high_ratio: float  # Low/High frequency ratio
    mid_high_ratio: float  # Mid/High frequency ratio
    
    # Spectral energy distribution
    spectral_entropy: float
    spectral_flatness: float
    spectral_rolloff: float
    
    # Directional features (for fibrous bands)
    horizontal_power: float
    vertical_power: float
    diagonal_power: float
    anisotropy_index: float  # Directional anisotropy
    
    # Peak features
    dominant_frequency: float
    frequency_peak_count: int
    peak_energy_concentration: float
    
    # NASH-specific features
    steatosis_frequency_signature: float  # Low-freq artifact from fat
    heterogeneity_index: float  # Texture heterogeneity measure
    
    # Advanced features
    phase_coherence: float
    magnitude_variance: float


class FFT2DAnalyzer:
    """
    2D Fast Fourier Transform Analyzer for Medical Imaging
    
    This class provides comprehensive FFT-based feature extraction
    specifically designed for liver fibrosis and NASH detection.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (512, 512),
                 window_function: str = 'hamming',
                 freq_bands: Dict[str, Tuple[float, float]] = None):
        """
        Initialize FFT Analyzer
        
        Parameters:
        -----------
        image_size : tuple
            Expected input image size (height, width)
        window_function : str
            Windowing function to reduce spectral leakage
            Options: 'hamming', 'hanning', 'blackman', 'bartlett', None
        freq_bands : dict
            Custom frequency band definitions
        """
        self.image_size = image_size
        self.window_function = window_function
        
        # Default frequency bands for fibrosis analysis
        if freq_bands is None:
            self.freq_bands = {
                'low': (0.0, 0.25),      # Overall structure
                'mid': (0.25, 0.75),     # Texture patterns
                'high': (0.75, 1.0)      # Fine details, fibrous bands
            }
        else:
            self.freq_bands = freq_bands
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before FFT
        
        Steps:
        1. Convert to grayscale if needed
        2. Resize to standard size
        3. Normalize intensity
        4. Apply windowing function
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (2D or 3D)
            
        Returns:
        --------
        preprocessed : np.ndarray
            Preprocessed image ready for FFT
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Enhance contrast (optional but recommended for CT)
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        
        # Apply windowing to reduce spectral leakage
        if self.window_function:
            window = self._create_2d_window(image.shape)
            image = image * window
        
        return image
    
    def _create_2d_window(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create 2D window function
        
        Parameters:
        -----------
        shape : tuple
            Shape of the window (height, width)
            
        Returns:
        --------
        window_2d : np.ndarray
            2D window function
        """
        if self.window_function == 'hamming':
            window_h = np.hamming(shape[0])
            window_w = np.hamming(shape[1])
        elif self.window_function == 'hanning':
            window_h = np.hanning(shape[0])
            window_w = np.hanning(shape[1])
        elif self.window_function == 'blackman':
            window_h = np.blackman(shape[0])
            window_w = np.blackman(shape[1])
        elif self.window_function == 'bartlett':
            window_h = np.bartlett(shape[0])
            window_w = np.bartlett(shape[1])
        else:
            return np.ones(shape)
        
        # Create 2D window by outer product
        window_2d = np.outer(window_h, window_w)
        return window_2d
    
    def compute_fft(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D FFT and return magnitude, phase, and power spectrum
        
        Parameters:
        -----------
        image : np.ndarray
            Preprocessed input image
            
        Returns:
        --------
        magnitude : np.ndarray
            Magnitude spectrum (abs of FFT)
        phase : np.ndarray
            Phase spectrum (angle of FFT)
        power_spectrum : np.ndarray
            Power spectrum (magnitude^2), shifted to center
        """
        # Compute 2D FFT
        fft_result = np.fft.fft2(image)
        
        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_result)
        
        # Compute magnitude and phase
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        # Compute power spectrum
        power_spectrum = magnitude ** 2
        
        # Log-scale for better visualization
        magnitude_log = np.log1p(magnitude)
        
        return magnitude_log, phase, power_spectrum
    
    def extract_frequency_band_power(self, power_spectrum: np.ndarray) -> Dict[str, float]:
        """
        Extract power in different frequency bands
        
        This is CRITICAL for fibrosis detection:
        - Low frequencies: Overall liver structure
        - Mid frequencies: Texture variations (NASH patterns)
        - High frequencies: Fibrous bands, cirrhotic nodules
        
        Parameters:
        -----------
        power_spectrum : np.ndarray
            2D power spectrum
            
        Returns:
        --------
        band_powers : dict
            Power in each frequency band
        """
        center_y, center_x = np.array(power_spectrum.shape) // 2
        max_radius = min(center_y, center_x)
        
        # Create radial frequency map
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        normalized_radius = radius / max_radius
        
        band_powers = {}
        for band_name, (r_min, r_max) in self.freq_bands.items():
            mask = (normalized_radius >= r_min) & (normalized_radius < r_max)
            band_power = np.sum(power_spectrum[mask])
            band_powers[band_name] = float(band_power)
        
        return band_powers
    
    def extract_directional_features(self, power_spectrum: np.ndarray) -> Dict[str, float]:
        """
        Extract directional frequency features
        
        Fibrous bands in cirrhosis often show directional patterns.
        This function analyzes power distribution in different directions.
        
        Parameters:
        -----------
        power_spectrum : np.ndarray
            2D power spectrum
            
        Returns:
        --------
        directional_features : dict
            Power in different directions
        """
        center_y, center_x = np.array(power_spectrum.shape) // 2
        
        # Define directional sectors (in radians)
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        angle = np.arctan2(y - center_y, x - center_x)
        
        # Horizontal: -π/8 to π/8 and 7π/8 to 9π/8
        horizontal_mask = ((np.abs(angle) < np.pi/8) | 
                          (np.abs(angle - np.pi) < np.pi/8))
        
        # Vertical: 3π/8 to 5π/8 and -5π/8 to -3π/8
        vertical_mask = ((np.abs(angle - np.pi/2) < np.pi/8) |
                        (np.abs(angle + np.pi/2) < np.pi/8))
        
        # Diagonal1: π/8 to 3π/8 and -7π/8 to -5π/8
        diagonal1_mask = (((angle > np.pi/8) & (angle < 3*np.pi/8)) |
                         ((angle > -7*np.pi/8) & (angle < -5*np.pi/8)))
        
        # Diagonal2: 5π/8 to 7π/8 and -3π/8 to -π/8
        diagonal2_mask = (((angle > 5*np.pi/8) & (angle < 7*np.pi/8)) |
                         ((angle > -3*np.pi/8) & (angle < -np.pi/8)))
        
        features = {
            'horizontal_power': float(np.sum(power_spectrum[horizontal_mask])),
            'vertical_power': float(np.sum(power_spectrum[vertical_mask])),
            'diagonal1_power': float(np.sum(power_spectrum[diagonal1_mask])),
            'diagonal2_power': float(np.sum(power_spectrum[diagonal2_mask]))
        }
        
        # Combine diagonals
        features['diagonal_power'] = features['diagonal1_power'] + features['diagonal2_power']
        
        # Anisotropy index (directional variability)
        powers = [features['horizontal_power'], features['vertical_power'], 
                 features['diagonal_power']]
        features['anisotropy_index'] = float(np.std(powers) / (np.mean(powers) + 1e-8))
        
        return features
    
    def compute_spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """
        Compute spectral entropy
        
        Measures randomness/disorder in frequency distribution.
        - High entropy: Heterogeneous texture (fibrosis, NASH)
        - Low entropy: Homogeneous tissue (healthy liver)
        
        Parameters:
        -----------
        power_spectrum : np.ndarray
            2D power spectrum
            
        Returns:
        --------
        entropy : float
            Spectral entropy value
        """
        # Normalize to probability distribution
        ps_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        
        # Remove zeros for log
        ps_normalized = ps_normalized[ps_normalized > 0]
        
        # Compute entropy
        entropy = -np.sum(ps_normalized * np.log2(ps_normalized + 1e-10))
        
        return float(entropy)
    
    def detect_dominant_frequency(self, power_spectrum: np.ndarray) -> Tuple[float, int]:
        """
        Detect dominant frequency and peak count
        
        Parameters:
        -----------
        power_spectrum : np.ndarray
            2D power spectrum
            
        Returns:
        --------
        dominant_freq : float
            Dominant frequency (normalized)
        peak_count : int
            Number of significant peaks
        """
        center_y, center_x = np.array(power_spectrum.shape) // 2
        
        # Exclude DC component (center)
        ps_no_dc = power_spectrum.copy()
        center_region = 5
        ps_no_dc[center_y-center_region:center_y+center_region,
                 center_x-center_region:center_x+center_region] = 0
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(ps_no_dc), ps_no_dc.shape)
        peak_y, peak_x = peak_idx
        
        # Compute radial distance (normalized frequency)
        dominant_freq = np.sqrt((peak_y - center_y)**2 + (peak_x - center_x)**2)
        dominant_freq /= min(center_y, center_x)
        
        # Count significant peaks (above threshold)
        threshold = 0.1 * np.max(ps_no_dc)
        peak_count = np.sum(ps_no_dc > threshold)
        
        return float(dominant_freq), int(peak_count)
    
    def compute_nash_frequency_signature(self, 
                                         power_spectrum: np.ndarray,
                                         band_powers: Dict[str, float]) -> float:
        """
        Compute NASH-specific frequency signature
        
        NASH (steatosis) creates:
        - Increased low-frequency power (fat deposits)
        - Reduced mid-frequency coherence (texture disruption)
        - Altered tissue heterogeneity
        
        Parameters:
        -----------
        power_spectrum : np.ndarray
            2D power spectrum
        band_powers : dict
            Power in different bands
            
        Returns:
        --------
        nash_signature : float
            NASH frequency signature score
        """
        # NASH indicator 1: Low/Mid frequency ratio
        low_mid_ratio = band_powers['low'] / (band_powers['mid'] + 1e-8)
        
        # NASH indicator 2: Total energy concentration in low frequencies
        total_power = sum(band_powers.values())
        low_concentration = band_powers['low'] / (total_power + 1e-8)
        
        # NASH indicator 3: Spectral irregularity
        ps_std = np.std(power_spectrum)
        ps_mean = np.mean(power_spectrum)
        irregularity = ps_std / (ps_mean + 1e-8)
        
        # Combined NASH signature (weighted)
        nash_signature = (0.4 * low_mid_ratio + 
                         0.4 * low_concentration + 
                         0.2 * irregularity)
        
        return float(nash_signature)
    
    def extract_all_features(self, image: np.ndarray) -> FFTFeatures:
        """
        Extract ALL FFT-based features from image
        
        This is the main function that orchestrates feature extraction.
        
        Parameters:
        -----------
        image : np.ndarray
            Input CT image
            
        Returns:
        --------
        features : FFTFeatures
            Complete set of FFT features
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Compute FFT
        magnitude, phase, power_spectrum = self.compute_fft(preprocessed)
        
        # Total power
        total_power = float(np.sum(power_spectrum))
        
        # Frequency band powers
        band_powers = self.extract_frequency_band_power(power_spectrum)
        
        # Frequency ratios (CRITICAL for fibrosis)
        low_high_ratio = band_powers['low'] / (band_powers['high'] + 1e-8)
        mid_high_ratio = band_powers['mid'] / (band_powers['high'] + 1e-8)
        
        # Spectral features
        spectral_entropy = self.compute_spectral_entropy(power_spectrum)
        spectral_flatness = float(np.exp(np.mean(np.log(power_spectrum + 1e-10))) / 
                                 (np.mean(power_spectrum) + 1e-10))
        spectral_rolloff = float(np.percentile(power_spectrum, 85))
        
        # Directional features
        directional_features = self.extract_directional_features(power_spectrum)
        
        # Dominant frequency and peaks
        dominant_freq, peak_count = self.detect_dominant_frequency(power_spectrum)
        peak_energy = float(np.max(power_spectrum) / (total_power + 1e-8))
        
        # NASH signature
        nash_signature = self.compute_nash_frequency_signature(power_spectrum, band_powers)
        
        # Heterogeneity index
        heterogeneity = float(np.std(power_spectrum) / (np.mean(power_spectrum) + 1e-8))
        
        # Phase coherence
        phase_coherence = float(np.abs(np.mean(np.exp(1j * phase))))
        
        # Magnitude variance
        magnitude_variance = float(np.var(magnitude))
        
        # Create feature object
        features = FFTFeatures(
            total_power=total_power,
            low_freq_power=band_powers['low'],
            mid_freq_power=band_powers['mid'],
            high_freq_power=band_powers['high'],
            low_high_ratio=low_high_ratio,
            mid_high_ratio=mid_high_ratio,
            spectral_entropy=spectral_entropy,
            spectral_flatness=spectral_flatness,
            spectral_rolloff=spectral_rolloff,
            horizontal_power=directional_features['horizontal_power'],
            vertical_power=directional_features['vertical_power'],
            diagonal_power=directional_features['diagonal_power'],
            anisotropy_index=directional_features['anisotropy_index'],
            dominant_frequency=dominant_freq,
            frequency_peak_count=peak_count,
            peak_energy_concentration=peak_energy,
            steatosis_frequency_signature=nash_signature,
            heterogeneity_index=heterogeneity,
            phase_coherence=phase_coherence,
            magnitude_variance=magnitude_variance
        )
        
        return features
    
    def visualize_fft(self, 
                     image: np.ndarray,
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> None:
        """
        Visualize FFT analysis results
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        save_path : str, optional
            Path to save figure
        show_plot : bool
            Whether to display plot
        """
        preprocessed = self.preprocess_image(image)
        magnitude, phase, power_spectrum = self.compute_fft(preprocessed)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original CT Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessed (Windowed)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Magnitude spectrum (log-scale)
        axes[0, 2].imshow(magnitude, cmap='hot')
        axes[0, 2].set_title('Magnitude Spectrum (Log)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Power spectrum
        axes[1, 0].imshow(np.log1p(power_spectrum), cmap='jet')
        axes[1, 0].set_title('Power Spectrum (Log)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Phase spectrum
        axes[1, 1].imshow(phase, cmap='hsv')
        axes[1, 1].set_title('Phase Spectrum', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Radial power profile
        center = np.array(power_spectrum.shape) // 2
        max_radius = min(center)
        y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        radial_profile = []
        for r in range(0, max_radius, 5):
            mask = (radius >= r) & (radius < r + 5)
            if np.any(mask):
                radial_profile.append(np.mean(power_spectrum[mask]))
        
        axes[1, 2].plot(radial_profile, linewidth=2, color='blue')
        axes[1, 2].set_title('Radial Power Profile', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Frequency (normalized)')
        axes[1, 2].set_ylabel('Average Power')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add vertical lines for frequency bands
        band_positions = [len(radial_profile) * 0.25, len(radial_profile) * 0.75]
        for pos in band_positions:
            axes[1, 2].axvline(x=pos, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("2D FFT Feature Extraction for Liver Fibrosis Staging")
    print("Projenin CAN ALICI NOKTASI: Frekans Domain Analizi")
    print("="*80)
    
    # Create dummy test image (simulating liver CT)
    test_image = np.random.randn(512, 512) * 50 + 100
    test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    
    # Initialize analyzer
    analyzer = FFT2DAnalyzer(window_function='hamming')
    
    # Extract features
    print("\nExtracting FFT features...")
    features = analyzer.extract_all_features(test_image)
    
    # Display features
    print("\n" + "="*80)
    print("EXTRACTED FFT FEATURES:")
    print("="*80)
    print(f"\n{'Feature Name':<40} {'Value':>15}")
    print("-"*80)
    
    for field, value in features.__dict__.items():
        print(f"{field:<40} {value:>15.6f}")
    
    print("\n" + "="*80)
    print("CRITICAL FIBROSIS INDICATORS:")
    print("="*80)
    print(f"Low/High Frequency Ratio: {features.low_high_ratio:.4f}")
    print(f"  → Higher ratio suggests more fibrosis (>2.0 suspicious)")
    print(f"\nNASH Frequency Signature: {features.steatosis_frequency_signature:.4f}")
    print(f"  → Higher value suggests NASH (>0.5 suspicious)")
    print(f"\nAnisotropy Index: {features.anisotropy_index:.4f}")
    print(f"  → Higher value indicates directional fibrosis (cirrhosis)")
    print(f"\nHeterogeneity Index: {features.heterogeneity_index:.4f}")
    print(f"  → Higher heterogeneity correlates with advanced fibrosis")
    
    print("\n" + "="*80)
    print("Module ready for integration!")
    print("="*80)
