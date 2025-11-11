# SarcGraph for 2DMB

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://doi.org/10.7910/DVN/GHMKWJ"><img src="https://img.shields.io/badge/Dataset-10.7910/DVN/GHMKWJ-blue" alt="Dataset"></a>
  <a href="https://github.com/Sarc-Graph/sarcgraph"><img src="https://img.shields.io/badge/Based%20on-SarcGraph-orange" alt="Sarc-Graph"></a>
</p>

SarcGraph-2DMB is an adaptation of the open-source Python package, [SarcGraph](https://github.com/Sarc-Graph/sarcgraph), specifically designed for the automated analysis of high-frame-rate recordings of two-dimensional cardiac muscle bundles (2DMBs). This project addresses the challenges of analyzing the larger and more physiologic sarcomere displacements and velocities observed in 2DMBs, for which previous sarcomere tracking pipeline has been unreliable.

## Project Overview

Timelapse images of human induced pluripotent stem cell-derived cardiomyocytes (hiPSC-CMs) provide rich information on cell structure and contractile function. The 2DMB platform standardizes tissue geometry, resulting in physiologic, uniaxial contractions of discrete tissues on an elastomeric substrate with stiffness similar to the heart. While 2DMBs are highly conducive to sarcomere imaging, their analysis has been challenging.

This adapted SarcGraph pipeline introduces several key modifications:

*   **Frame-by-frame sarcomere detection:** Automated tissue segmentation with spatial partitioning.
*   **Gaussian Process Regression:** For robust signal denoising of sarcomere length signals, with GPU acceleration using GPyTorch.
*   **Automated contractile phase detection:** A robust framework to identify the contraction phase of the cell.
*   **Tissue partitioning:** For more granular analysis of sarcomeric features.

These enhancements enable the extraction of structural organization and functional contractility metrics for both the whole 2DMB tissue and distinct tissue regions in a fully automated manner.

This repository contains the source code for the adapted SarcGraph pipeline, along with several Jupyter notebook tutorials to guide users through the analysis. The accompanying dataset of 130 example movies of baseline and drug-treated samples is available on the [Harvard Dataverse](httpss://doi.org/10.7910/DVN/GHMKWJ).

## Installation

To use this adapted version of SarcGraph, follow these steps:

1.  First, create an environment and install the base `sarcgraph` package by following the instructions on the official documentation: [https://sarc-graph.readthedocs.io/en/latest/installation.html#installation-ref](https://sarc-graph.readthedocs.io/en/latest/installation.html#installation-ref)

2.  Once `sarcgraph` is installed in your environment, you can install the additional dependencies required for this project by running the following command in your terminal:

    ```bash
    pip install -r requirements.txt
    ```

## Tutorials

This repository includes a series of Jupyter notebooks to guide you through the entire analysis pipeline. Each notebook covers a specific module of the pipeline:

*   **01_segmentation_module_tutorial.ipynb:** Demonstrates how to perform tissue segmentation.
*   **02_detection_module_tutorial.ipynb:** Shows how to use SarcGraph to detect sarcomeres and Z-discs in each frame of the video.
*   **03_denoising_module_tutorial.ipynb:** Explains how to use Gaussian Process Regression (GPR) to denoise the raw sarcomere length signals.
*   **04_analyze_module_tutorial.ipynb:** Shows how to process the denoised sarcomere length signal to detect contractile phases.
*   **05_feature_extraction_module_tutorial.ipynb:** Shows how to extract a comprehensive set of functional and structural metrics from the analyzed data.
*   **06_partition_module_tutorial.ipynb:** Demonstrates how to partition the tissue into different regions for more granular analysis.
*   **full_pipeline_tutorial.ipynb:** A comprehensive notebook that runs the entire pipeline from start to finish.

## Features and Metrics

For a detailed explanation of each extracted feature and metric, please refer to the [`metrics_guide.md`](metrics_guide.md) file.

## Dataset

The dataset accompanying this project is available on the Harvard Dataverse and consists of 130 sample videos recorded from 65 tissues in both baseline and drug-treated conditions. The drug treatments include:

*   20 samples treated with Mavacamten
*   25 samples treated with Endothelin-1
*   20 samples treated with Isoproterenol

All videos are compressed using 7z to meet the file size requirements of the Harvard Dataverse. After decompression, the videos are in the `.nd2` format.

You can download the dataset:

1.  **Through the Harvard Dataverse UI:** [https://doi.org/10.7910/DVN/GHMKWJ](https://doi.org/10.7910/DVN/GHMKWJ)
2.  **Using the provided Python script:** We have included a script that demonstrates how to download the dataset programmatically.

## Contact

For questions or issues, please open an issue on this GitHub repository or contact Saeed Mohammadzadeh at [saeedmhz@bu.edu](mailto:saeedmhz@bu.edu).