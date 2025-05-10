#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# Stand‑alone Streamlit front‑end for a multimodal AudioCLIP‑based
# emotion detector.  The file includes tiny helpers so it works even
# when no other project modules are present.
# ----------------------------------------------------------------------
# Requires   :  streamlit, torch, torchvision, torchaudio, matplotlib,
#               pillow, librosa, numpy
# Model file :  emotion_detector_audioclip.pth  (state‑dict or scripted)
# ----------------------------------------------------------------------
# Run        :  streamlit run app.py
# ──────────────────────────────────────────────────────────────────────

import io
import math
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchaudio
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchaudio
from torchaudio import transforms as audio_transforms
from transformers import CLIPVisionModel          # instead of torchvision resnet for vision branch
from torch.cuda.amp import autocast, GradScaler   # mixed‑precision training
from torch.optim import AdamW

# ────────────────────────── global settings ───────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "emotion_detector_audioclip.pth"  # change if needed
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMG_SIZE = 224
IMAGE_TFMS = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # images are grayscale / 3‑ch OK
    ]
)

# ────────────────────────── model helpers ─────────────────────────────
@st.cache_resource(show_spinner=True)

class AudioCLIPEmotionDetector(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=len(EMOTIONS)):  # Changed to 512 to match CLIP
        super(AudioCLIPEmotionDetector, self).__init__()
        
        # Text encoder from CLIP (frozen)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Image encoder from CLIP (we'll fine-tune this)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = models.resnet50(pretrained=True)
        # Replace the final layer
        self.image_encoder.fc = nn.Linear(2048, embedding_dim)
        
        # Audio encoder (ESResNeXt)
        self.audio_encoder = ESResNeXt(embedding_dim=embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.vision_encoder.parameters():
            p.requires_grad = False              # freeze backbone for stability
        self.image_proj = nn.Linear(
            self.vision_encoder.config.hidden_size, embedding_dim
            )
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def _get_emotion_text_embeddings(self):
        """Create text embeddings for each emotion class"""
        emotion_prompts = [f"a {emotion} expression" for emotion in EMOTIONS]
        
        # Tokenize emotions
        inputs = self.tokenizer(emotion_prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get text embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_embeddings = outputs.pooler_output
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        return text_embeddings
    
    # def encode_image(self, images):
    #     """Encode images to embedding space"""
    #     embeddings = self.image_encoder(images)
    #     return F.normalize(embeddings, p=2, dim=1)

    def encode_image(self, images):
        
        with torch.no_grad():                    # backbone frozen
            vision_out = self.vision_encoder(pixel_values=images).pooler_output
        embeddings = self.image_proj(vision_out)
        return F.normalize(embeddings, p=2, dim=1)
    
    def encode_audio(self, audio_specs):
        """Encode audio spectrograms to embedding space"""
        return self.audio_encoder(audio_specs)
    
    def forward(self, x, modality='image'):
        """Forward pass based on modality"""
        if modality == 'image':
            embeddings = self.encode_image(x)
        elif modality == 'audio':
            embeddings = self.encode_audio(x)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Generate emotion text embeddings on the fly (fixes the shape issue)
        emotion_text_embeddings = self._get_emotion_text_embeddings()
        
        # Compute similarity with emotion text embeddings
        similarity = embeddings @ emotion_text_embeddings.T
        
        # Classification logits
        logits = self.classifier(embeddings)
        
        return logits, similarity, embeddings
        
def load_model(path="emotion_detector_audioclip.pth"):
    model = AudioCLIPEmotionDetector().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# def load_model(weights: str = WEIGHTS_PATH):
#     """
#     Load a fine‑tuned AudioCLIPEmo model.
#     This function is intentionally minimal: it assumes the *entire* model
#     (architecture + weights) was saved with `torch.save(model)`.
#     If you only saved state_dict, adapt as needed.
#     """
#     if not Path(weights).is_file():
#         st.error(
#             f"Model file '{weights}' not found. "
#             f"Please put your trained weights in the working directory."
#         )
#         st.stop()

#     model = torch.load(weights, map_location=DEVICE)
#     model.eval()
#     return model


def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _audio_to_melspec_img(path: str, sr_target: int = 16000) -> torch.Tensor:
    """Load audio file → mel‑spectrogram → 3‑channel torch.Tensor."""
    waveform, sr = torchaudio.load(path)
    if sr != sr_target:
        resampler = torchaudio.transforms.Resample(sr, sr_target)
        waveform = resampler(waveform)
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 128‑bin mel spec
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr_target, n_mels=128, n_fft=1024, hop_length=512
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
    mel_img = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img = mel_img.expand(3, -1, -1)  # to 3‑channel
    # resize to match IMG_SIZE
    mel_img = torch.nn.functional.interpolate(
        mel_img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0)
    return mel_img


@torch.inference_mode()
def predict_emotion(model, inp, modality: str):
    """
    Return:
        pred  : str  (top‑1 emotion name)
        probs : list float  (len == len(EMOTIONS))
    """
    if modality == "image":
        if isinstance(inp, Image.Image):
            tensor = IMAGE_TFMS(inp).unsqueeze(0).to(DEVICE)
        else:
            raise ValueError("For image modality, inp must be PIL Image")
    elif modality == "audio":
        tensor = _audio_to_melspec_img(str(inp)).unsqueeze(0).to(DEVICE)
    else:
        raise ValueError("modality must be 'image' or 'audio'")

    # IMPORTANT: adapt to your model’s forward signature!
    logits = model(tensor) if hasattr(model, "__call__") else model.forward(tensor)
    probs = _logits_to_probs(logits).squeeze().cpu().numpy().tolist()
    pred_idx = int(np.argmax(probs))
    return EMOTIONS[pred_idx], probs


def _make_bar(probabilities):
    """Return a matplotlib figure with emotion‑probability bars."""
    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.array(probabilities)
    ax.bar(range(len(EMOTIONS)), y)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=40, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Emotion probabilities")
    fig.tight_layout()
    return fig


# ────────────────────────── Streamlit UI ──────────────────────────────
st.set_page_config(page_title="Emotion Detector (AudioCLIP)", layout="wide")
st.title("🎭 Multimodal Emotion Detector (AudioCLIP)")
st.markdown(
    "Upload a **face image** or a **speech clip** and the model will guess the emotion."
)

model = load_model()

tab_img, tab_audio = st.tabs(["🖼️ Image", "🔊 Audio"])

# ----------------- image tab -----------------
with tab_img:
    img_file = st.file_uploader(
        "JPEG / PNG image of a face", type=["jpg", "jpeg", "png"]
    )
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Input image", use_column_width=False)
        pred, probs = predict_emotion(model, img, modality="image")
        st.success(f"**Prediction:** {pred.capitalize()}")
        st.pyplot(_make_bar(probs))

# ----------------- audio tab -----------------
with tab_audio:
    audio_file = st.file_uploader(
        "WAV / MP3 / FLAC speech clip (≤ 15 s recommended)",
        type=["wav", "mp3", "flac"],
    )
    if audio_file:
        # write upload to a NamedTemporaryFile for torchaudio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path)
        pred, probs = predict_emotion(model, tmp_path, modality="audio")
        st.success(f"**Prediction:** {pred.capitalize()}")
        st.pyplot(_make_bar(probs))

st.caption(
    "Weights must be saved as **emotion_detector_audioclip.pth** "
    "in the same folder as this file."
)
