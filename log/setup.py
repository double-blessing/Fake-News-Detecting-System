from setuptools import setup
import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected
build_exe_options = {
    "packages": ["flask", "joblib", "spacy", "requests", "sklearn"],
    "includes": ["en_core_web_sm"],
    "include_files": [
        ("templates/", "templates/"),
        ("model/", "model/"),
        ("utils/", "utils/"),
        ("log/", "log/"),
        # Include spaCy data directory
        ("venv/Lib/site-packages/en_core_web_sm", "en_core_web_sm")
    ]
}

# GUI applications require a different base on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="FakeNewsDetector",
    version="1.0",
    description="Fake News Detection Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("app.py", base=base, target_name="FakeNewsDetector.exe")]
)