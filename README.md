# Fake-Image-Forensics-Detector
This project provides a computer vision-based forensic system that identifies whether an image is real, AI-generated, or AI-edited. The system supports models such as UnivFD, PatchCraft, and DIRE, along with evaluation tools and catastrophic generalization testing.

## Features
- Multi-model architecture for forensic classification
- Reconstruction-based detection using DIRE-style methods
- Augmentation and preprocessing modules
- Bucket-wise evaluation: real, AI-generated, AI-edited
- ROC, AUC, precision, recall and confusion matrix reporting
- Dataset loader and metadata generator
- REST API for inference

## Architecture Overview
1. Image preprocessing and normalization
2. Model selection (UnivFD, PatchCraft, DIRE or ensemble)
3. Feature extraction and classification
4. Evaluation using multiple metrics
5. Optional ensemble scoring
6. API for inference with image uploads

See docs/architecture.md and docs/model-comparison.md for more information.

## Project Structure
- models: Pretrained and custom implementations
- src: Inference, preprocessing, evaluation and utilities
- data: Image datasets and metadata
- docs: Design and research notes
- api: REST server and OpenAPI spec
- notebooks: Experiments and analysis
- tests: Automated tests
