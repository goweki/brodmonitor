# Asset Tagging and Recognition Model for Broadcast Monitoring

This project is part of a larger broadcast monitoring framework that aims to help tag and recognize audio assets in broadcast media, such as TV shows, news clips, and commercials. The goal is to enable broadcasters and advertisers to quickly and accurately identify and track their assets across different channels and media platforms.

This project uses Jupyter Notebook to break down the ste-by-step process as will be implemented in the framework to show the inner working s of the framework.

The asset tagging and recognition model is built using Python and Jupyter notebooks, and will utilize various open-source algorithms to analyze and classify broadcast media content. The model will also interact with ML models trained on a large dataset of labeled assets, and is designed to work with different types of media formats and sources.

## Requirements

To run the asset tagging and recognition model, you will need to have the following installed:

- Python 3.6 or higher
- Jupyter Notebook
- NumPy
- Matplotlib
- FFmpeg
- pydub

## Getting Started

To get started with the asset tagging and recognition model, you can follow these steps:

- Clone the repository to your local machine.
- I recommend installing the required packages within a virtual environment.
- Copy an audio segment that will be processed for watermarking and/or fingerprinting into the same directory as the Jupyter notebook.
- Open the Jupyter notebook bmonitor.ipynb and in the first cell, replace **Sanchez - Mama (African Pride).mp3** with the name of the audio file you copied into the directory in the previous step.
- Run the first notebook cells to load the audio file you copied into the directory.
- Run the consecutive cells to fingerprint and watermark the audio file.
- Having generated a fingerprint and/or watermarked the audio, youmay run inferences on broadcasts to detect if the audio segment you processed may be within the audio stream.

## Usage

Free to use
