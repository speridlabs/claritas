from setuptools import setup, find_packages

setup(
    name="claritas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "tqdm>=4.60.0",
    ],
    author="Sperid Labs",
    author_email="contact@speridlabs.com",
    description="A library for selecting and managing sharp images from videos or image collections",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/speridlabs/claritas",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "claritas=claritas.cli:main",
        ],
    },
)

