import shutil
from setuptools import setup, find_packages

def check_dependencies():
    """Check if required dependencies are installed and accessible."""
    missing = []
    if shutil.which('ffmpeg') is None:
        missing.append("ffmpeg")
    if shutil.which('exiftool') is None:
        missing.append("exiftool")
        
    if missing:
        tools = ", ".join(missing)
        raise RuntimeError(
            f"{tools} not found. Please install required tools first.\n"
            "On Ubuntu/Debian: sudo apt-get install ffmpeg libimage-exiftool-perl\n"
            "On MacOS: brew install ffmpeg exiftool\n"
            "On Windows: Download ffmpeg from https://www.ffmpeg.org/download.html and exiftool from https://exiftool.org/"
        )

check_dependencies()

setup(
    name="claritas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "tqdm>=4.60.0",
        "pycolmap>=3.13.0",
        "matplotlib>=3.5.0",
    ],
    author="Sperid Labs",
    author_email="contact@speridlabs.com",
    description="A library for selecting and managing sharp images from videos or image collections",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/speridlabs/claritas",
        "Source Code": "https://github.com/speridlabs/claritas",
        "Issues": "https://github.com/speridlabs/claritas/issues",
    },
    url="https://github.com/speridlabs/claritas",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "claritas=claritas.cli:main",
        ],
    },
)

