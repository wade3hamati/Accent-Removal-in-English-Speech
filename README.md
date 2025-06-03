# Accent Removal Project

## Project Overview

This project explores accent removal in English speech while preserving speaker identity, addressing the challenge of limited paired accented and non-accented speech samples from the same speaker. The system uses a dataset of Indian-accented English recordings and synthetic American-accented versions of the same sentences for training.

## Approach

The project implements a two-stage approach:

1. **Wav2Vec2-based Classifier**: Distinguishes accented from non-accented speech
2. **EnCodec-based Generator**: Guided by the classifier to produce accent-neutralized speech

Speaker identity preservation is enforced using ECAPA-TDNN embeddings. An alternative method involving SpeechTokenizer's codebook manipulation was attempted but proved ineffective.

## Key Components

- **Accent Classifier**: Conv-frozen Wav2Vec2 model (wav2vec2-base-960h variant) with MLP classifier
- **SpeechTokenizer**: 8-layer vector-quantized variational autoencoder (VQ-VAE) architecture
  - Codebook 1: Accent Modeling (MLP head trained with negative log likelihood loss)
  - Codebook 2: Speaker Preservation (parallel MLP head trained with negative log likelihood)
- **Adversarial Feedback Loop**: Frozen discriminator backpropagates through decoder
- **Speaker Consistency**: ECAPA-TDNN embeddings compared using cosine similarity loss
- **Audio Reconstruction**: MSE loss on mel-spectrograms ensures output quality

## Results

Preliminary results suggest the discriminator-guided approach shows promise in reducing accent cues while maintaining speaker similarity, though challenges remain in achieving naturalness and complete accent removal.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- HuggingFace Transformers
- Other dependencies (see requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download pretrained models (Wav2Vec2, ECAPA-TDNN, SpeechTokenizer)

## Dataset

The model is trained on:
- Indian-accented English recordings
- Synthetic American-accented versions of the same sentences

## Future Work

- Improved disentanglement techniques between accent and speaker attributes
- Enhanced naturalness of output speech
- Complete accent removal
