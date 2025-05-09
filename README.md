# Emotion Detector with AudioCLIP  
*by **Hardik Pahwa***

---

## ✨ Motivation  
Most emotion‑recognition systems handle **either** faces **or** voices.  
This project tries both at once. Using **AudioCLIP**—a version of OpenAI’s  
**CLIP** that also understands sound—the model learns to tell whether a person  
looks **and** sounds happy, sad, angry, etc.

---

##  Where It Fits in Multimodal Learning  

| Year | Milestone | Why it matters |
|------|-----------|----------------|
| 2021 | **CLIP** – links images and text | Large‑scale contrastive learning → strong, reusable features |
| 2021 | **AudioCLIP** – adds an audio branch to CLIP | One embedding space for **text ,  image , audio** |
| 2023‑24 | Fine‑tuning AudioCLIP for niche tasks | Works well with small datasets |
| **This repo** | Fine‑tunes AudioCLIP on **FER‑2013** (faces) + **CREMA‑D** (speech) | Tests tri‑modal emotion recognition |

---

## 🗒️ Project Overview  

* **Datasets**  
  * **FER‑2013** – 48×48 grayscale face images, 7 emotions  
  * **CREMA‑D** – audio clips of actors speaking with 6 emotions  

* **Pipeline**  
  1. Convert WAV clips to mel‑spectrogram “images”.  
  2. Feed face images and spectrograms through the same vision backbone.  
  3. Use a contrastive loss so matching face/voice pairs stay close in the embedding space.  
  4. Add a small classifier head to predict the emotion label.

* **Training tricks** (see `jupyter notebook`)  
  * Mixed‑precision (`torch.cuda.amp`)  
  * `AdamW` optimizer with cosine scheduler  
  * Early stopping on validation loss

---

## Quick Start  

```bash
# 1. Clone
git clone https://github.com/your‑handle/emotion‑detector‑audioclip.git
cd emotion‑detector‑audioclip

# 2. (Optional) create virtual env
python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Put the datasets
#    data/fer2013/train/<emotion>/*.png
#    data/cremad/AudioWAV/*.wav

# 5. Train
python audio_clip.py                     # 20 epochs by default

# 6. Streamlit demo
streamlit run app.py

```

## What I Learned
Converting audio to 2‑D spectrograms lets one backbone handle both views.

A simple contrastive loss (image vs. audio) boosts performance by > 3 %.


## Reflections & Future Work
---
|                  | Notes                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| **Surprises**    | Fear and surprise look alike in low‑res faces; voice helps separate them.                         |
| **Limitations**  | Audio clips are cropped to 5 s, missing long expressions.                                         |
| **Improvements** | Add video frames, use a larger ViT backbone, or train with dynamic time warping for longer audio. |
---

# References
A. Guzhov et al., “AudioCLIP: Extending CLIP to Image, Text and Audio,” 2021.

GitHub: https://github.com/AndreyGuzhov/AudioCLIP

FER‑2013 dataset (Kaggle).

CREMA‑D dataset (Kaggle).

