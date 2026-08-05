from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cctv-agent",
    version="1.0.0",
    description="Distributed CCTV edge processing agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python",
        "numpy",
        "psutil",
        "supervision>=0.27.0",  
        "ultralytics>=8.4.0",    
        "pyyaml",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "cctv-agent=cctv_agent.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)