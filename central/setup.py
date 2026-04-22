from setuptools import setup, find_packages

with open("../README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cctv-central",
    version="1.0.0",
    description="Central server for distributed CCTV monitoring system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi==0.135.1",
        "uvicorn[standard]==0.32.1",
        "python-multipart==0.0.9",
    ],
    entry_points={
        "console_scripts": [
            "cctv-central=central.server:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)