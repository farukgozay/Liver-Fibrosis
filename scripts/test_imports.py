"""
Test Import - Dependency Kontrolü
=================================

Tüm gerekli paketlerin kurulu olup olmadığını test eder.
"""

import sys

def test_imports():
    """Test all required imports"""
    
    print("Testing imports...")
    print("=" * 50)
    
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib.pyplot"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("shap", "shap"),
        ("pydicom", "pydicom"),
        ("SimpleITK", "SimpleITK"),
        ("cv2", "opencv-python"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
    ]
    
    failed = []
    
    for package_name, install_name in required_packages:
        try:
            __import__(package_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - MISSING!")
            failed.append(install_name)
    
    print("=" * 50)
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) missing!")
        print("\nInstall with:")
        print(f"pip install {' '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✅ All imports successful!")
        print("You are ready to run the pipeline!")
        sys.exit(0)

if __name__ == "__main__":
    test_imports()
