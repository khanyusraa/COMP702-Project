from cx_Freeze import setup, Executable
import os
# Include your model and JSON
include_files = [
    ("best_hybrid_model.keras", "best_hybrid_model.keras"),
    ("class_indices.json", "class_indices.json"),
    ("vcomp140.dll", "vcomp140.dll"),
    ("msvcp140.dll", "msvcp140.dll"),
    ("vcruntime140.dll", "vcruntime140.dll"),
    ("vcruntime140_1.dll", "vcruntime140_1.dll"),
    ("concrt140.dll", "concrt140.dll")
]


# Add environment fixes for TensorFlow DLL loading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

setup(
    name="CoinClassifier",
    version="1.0",
    description="Coin GUI Classifier",
    options={
        "build_exe": {
            "includes": ["tkinter", "tensorflow", "cv2", "PIL.ImageTk"],
            "include_files": include_files,
            "packages": ["numpy", "tensorflow", "cv2", "PIL", "tkinter"]
        }
    },
    executables=[
        Executable("coinGUI.py", base="Win32GUI")  # No console
    ]
)
