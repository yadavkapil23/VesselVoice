import os
import sys
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import torch
import torch.nn as nn
import torchvision.models as models


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths to saved models
CNN_WEIGHTS    = os.path.join(PROJECT_ROOT, "Classification models", "Deep learning  CNN on mel log spectrogram (MFCC)", "model weights", "cnn_scratch.pth")
RESNET_WEIGHTS = os.path.join(PROJECT_ROOT, "Classification models", "Deep learning  CNN on mel log spectrogram (MFCC)", "model weights", "resnet18_finetuned.pth")
KNN_MODEL      = os.path.join(PROJECT_ROOT, "Classification models", "Baseline classical classifier models", "models", "knn_best.pkl")
SVM_MODEL      = os.path.join(PROJECT_ROOT, "Classification models", "Baseline classical classifier models", "models", "svm_rbf_best.pkl")
RF_MODEL       = os.path.join(PROJECT_ROOT, "Classification models", "Baseline classical classifier models", "models", "rf_best.pkl")
SCALER_PATH    = os.path.join(PROJECT_ROOT, "Feature extraction", "Hand crafted features", "scaler.pkl")

TARGET_SR       = 16000   
SEGMENT_DURATION = 5.0    
N_MELS          = 128     
N_FFT           = 400     
HOP_LENGTH      = 160     
SPEC_SIZE       = 128     
N_SAMPLES       = int(SEGMENT_DURATION * TARGET_SR) # 80000

# Class mapping
CLASS_NAMES = [
    'Motorboats (A)',
    'Mussel Boats (B)',
    'Fishing Vessels (C)',
    'Passengers/Ferries (D)',
    'Ocean Liners/Tugboats (E)',
]
N_CLASSES = 5



class ShipsEarCNN(nn.Module):
    """
    Custom 3-block CNN for ShipsEar spectrogram classification.
    Architecture:
      3× [Conv2D → BatchNorm → ReLU → MaxPool(2×2)]
      → GlobalAvgPool → Dense(256) → Dropout(0.4) → Dense(5)
    Input : (B, 1, 128, 128)
    Output: (B, 5) logits
    """
    def __init__(self, n_classes=5, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

class ResNet18ShipsEar(nn.Module):
    """
    ResNet-18 fine-tuned for ShipsEar.
    Input: (B, 1, 128, 128) single-channel spectrogram.
    Repeats channel to RGB: (B, 3, 128, 128) for ImageNet compatibility.
    Final FC replaced with Dense(5).
    """
    def __init__(self, n_classes=5):
        super().__init__()
        # Use weights=None since we are loading our own finetuned weights
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, n_classes)
        self.backbone = backbone

    def forward(self, x):
        # Repeat single channel to 3 channels to match ImageNet input
        # x is (B, 1, H, W) -> (B, 3, H, W)
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)

def load_and_preprocess_audio(audio_path, target_sr=TARGET_SR, segment_duration=SEGMENT_DURATION):
    """
    Load an audio file, resample to target_sr, and split into
    fixed-length segments (zero-pad last segment if needed).
    Returns a list of audio segments (numpy arrays).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    segment_samples = int(segment_duration * target_sr)
    segments = []

    for start in range(0, len(y), segment_samples):
        seg = y[start:start + segment_samples]

        if len(seg) < segment_samples:
            seg = np.pad(seg, (0, segment_samples - len(seg)), mode='constant')
        segments.append(seg)

    return segments, sr


def extract_spectrogram(segment, sr=TARGET_SR):
    """
    Extract a log-mel spectrogram from an audio segment.
    Returns a (1, 128, 128) torch tensor suitable for the CNN/ResNet models.
    Normalization and resizing match the training notebook exactly.
    """
    from PIL import Image
    # Log-mel spectrogram
    S = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=sr // 2
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # range: [-80, 0]

    # Map [-80, 0] dB to [0, 1] then to [0, 255] for PIL
    S_norm = (S_db + 80.0) / 80.0
    S_uint = (S_norm * 255).clip(0, 255).astype(np.uint8)

    # Resize to 128x128 using Bilinear interpolation
    img = Image.fromarray(S_uint).resize((SPEC_SIZE, SPEC_SIZE), Image.BILINEAR)
    spec = np.array(img, dtype=np.float32)

    # Map back to dB range
    spec = (spec / 255.0) * 80.0 - 80.0

    return torch.from_numpy(spec).unsqueeze(0)


def extract_mfcc_features(y, sr=TARGET_SR):
    """40 MFCCs + delta + delta-delta (240-dim)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = []
    for mat in [mfcc, delta, delta2]:
        feats.extend(np.mean(mat, axis=1).tolist())
        feats.extend(np.std(mat, axis=1).tolist())
    return np.array(feats, dtype=np.float32)

def extract_spectral_features(y, sr=TARGET_SR):
    """Spectral features matching Part 4 notebook (24-dim)."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)

    feats = []
    for mat in [centroid, bandwidth, rolloff, flatness]:
        feats.append(np.mean(mat)); feats.append(np.std(mat))
    feats.extend(np.mean(contrast, axis=1).tolist())
    feats.extend(np.std(contrast, axis=1).tolist())
    # RMS and ZCR (mean only, as in notebook)
    feats.append(np.mean(rms)); feats.append(np.mean(zcr))
    return np.array(feats[:24], dtype=np.float32)

def extract_stft_features(y, sr=TARGET_SR):
    """STFT magnitude statistics over 20 bands (40-dim)."""
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann'))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    n_bins, n_bands = D_db.shape[0], 20
    band_means, band_stds = [], []
    indices = np.array_split(np.arange(n_bins), n_bands)
    for idx in indices:
        band_slice = D_db[idx, :]
        band_means.append(np.mean(band_slice)); band_stds.append(np.std(band_slice))
    return np.array(band_means + band_stds, dtype=np.float32)

def extract_lofar_features(y, sr=TARGET_SR, top_k=10):
    """LOFAR top-K spectral peaks (20-dim)."""
    from scipy import signal
    from scipy.signal import find_peaks
    freqs, psd = signal.welch(y, fs=sr, nperseg=4096, noverlap=2048, window='hann')
    psd_db = 10 * np.log10(psd + 1e-12)
    peaks, _ = find_peaks(psd_db, height=np.percentile(psd_db, 75), distance=5)
    peak_freqs = np.zeros(top_k, dtype=np.float32)
    peak_amps = np.zeros(top_k, dtype=np.float32)
    if len(peaks) > 0:
        sort_idx = np.argsort(psd_db[peaks])[::-1]
        n = min(top_k, len(peaks))
        peak_freqs[:n] = freqs[peaks[sort_idx[:n]]]
        peak_amps[:n] = psd_db[peaks[sort_idx[:n]]]
    return np.concatenate([peak_freqs / (sr / 2.0), peak_amps]).astype(np.float32)

def extract_demon_features(y, sr=TARGET_SR, n_peaks=5):
    """DEMON envelope modulation peaks (10-dim)."""
    from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch
    nyq = sr / 2.0
    b, a = butter(4, [100 / nyq, 5000 / nyq], btype='band')
    y_bp = filtfilt(b, a, y)
    envelope = np.abs(hilbert(y_bp))
    freqs, psd = welch(envelope, fs=sr, nperseg=min(4096, len(envelope)//4), window='hann')
    psd_db = 10 * np.log10(psd + 1e-12)
    mask = (freqs >= 0.5) & (freqs <= 50)
    f_sub, p_sub = freqs[mask], psd_db[mask]
    peaks, _ = find_peaks(p_sub, height=np.percentile(p_sub, 60), distance=3)
    pf, pa = np.zeros(n_peaks, dtype=np.float32), np.zeros(n_peaks, dtype=np.float32)
    if len(peaks) > 0:
        sort_idx = np.argsort(p_sub[peaks])[::-1]
        n = min(n_peaks, len(peaks))
        pf[:n] = f_sub[peaks[sort_idx[:n]]]; pa[:n] = p_sub[peaks[sort_idx[:n]]]
    return np.concatenate([pf / 50.0, pa]).astype(np.float32)

def extract_handcrafted_features(segment, sr=TARGET_SR):
    """
    Extract full 334-dimensional handcrafted feature vector matching the training pipeline.
    """
    f1 = extract_mfcc_features(segment, sr)     # 240
    f2 = extract_spectral_features(segment, sr) # 24
    f3 = extract_stft_features(segment, sr)     # 40
    f4 = extract_lofar_features(segment, sr)    # 20
    f5 = extract_demon_features(segment, sr)    # 10
    return np.concatenate([f1, f2, f3, f4, f5]).astype(np.float32)


def load_cnn_model(device='cpu'):
    """Load the custom CNN model with saved weights."""
    model = ShipsEarCNN(n_classes=N_CLASSES)
    state_dict = torch.load(CNN_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_resnet_model(device='cpu'):
    """Load the fine-tuned ResNet-18 model."""
    model = ResNet18ShipsEar(n_classes=N_CLASSES)
    
    if not os.path.isfile(RESNET_WEIGHTS):
        raise FileNotFoundError(f"ResNet weights not found: {RESNET_WEIGHTS}")

    state_dict = torch.load(RESNET_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_classical_model(model_type='svm'):
    """Load a classical ML model (SVM, KNN, or RF) and the scaler."""
    model_paths = {
        'svm': SVM_MODEL,
        'knn': KNN_MODEL,
        'rf':  RF_MODEL,
    }

    if model_type not in model_paths:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_paths.keys())}")

    model_path = model_paths[model_type]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load scaler
    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

def predict_with_dl_model(model, spectrograms, device='cpu'):
    """
    Run prediction using a deep learning model (CNN or ResNet).
    Returns class predictions and probability distributions for each segment.
    """
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for spec in spectrograms:
            X = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 128, 128)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            all_preds.append(pred)
            all_probs.append(probs.cpu().numpy().flatten())

    return all_preds, all_probs


def predict_with_classical_model(model, scaler, features):
    """
    Run prediction using a classical ML model.
    Returns class predictions and probability distributions for each segment.
    """
    all_preds = []
    all_probs = []

    for feat in features:
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        pred = model.predict(feat_scaled)[0]
        all_preds.append(int(pred))

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            prob_raw = model.predict_proba(feat_scaled).flatten()
            # Handle cases where model.classes_ might not contain all N_CLASSES
            if len(prob_raw) == N_CLASSES:
                prob = prob_raw
            else:
                prob = np.zeros(N_CLASSES)
                for i, cls_idx in enumerate(model.classes_):
                    if cls_idx < N_CLASSES:
                        prob[int(cls_idx)] = prob_raw[i]
            all_probs.append(prob)
        else:
            # One-hot encode prediction as pseudo-probability
            prob = np.zeros(N_CLASSES)
            if int(pred) < N_CLASSES:
                prob[int(pred)] = 1.0
            all_probs.append(prob)

    return all_preds, all_probs

def predict(audio_path, model_type='cnn', top_k=3, device='cpu'):
    """
    Full prediction pipeline:
    1. Load and preprocess audio
    2. Extract features/spectrograms
    3. Run model inference
    4. Aggregate results across segments
    5. Return final prediction with confidence scores
    """
    print(f"\n{'='*60}")
    print(f"  VesselVoice Classifier")
    print(f"{'='*60}")
    print(f"  Audio file : {audio_path}")
    print(f"  Model      : {model_type.upper()}")
    print(f"{'='*60}\n")

    # Step 1: Load audio
    print("[1/4] Loading and preprocessing audio...")
    segments, sr = load_and_preprocess_audio(audio_path)
    duration = len(segments) * SEGMENT_DURATION
    print(f"      Sample rate: {sr} Hz")
    print(f"      Duration: {duration:.1f}s -> {len(segments)} segment(s)\n")

    # Step 2: Extract features / spectrograms
    print("[2/4] Extracting features...")

    if model_type in ('cnn', 'resnet'):
        spectrograms = [extract_spectrogram(seg, sr) for seg in segments]
        print(f"      Spectrogram shape per segment: {spectrograms[0].shape}\n")
    else:
        features = [extract_handcrafted_features(seg, sr) for seg in segments]
        print(f"      Feature vector length: {len(features[0])}\n")

    # Step 3: Run inference
    print(f"[3/4] Running {model_type.upper()} model inference...")

    if model_type == 'cnn':
        model = load_cnn_model(device)
        preds, probs = predict_with_dl_model(model, spectrograms, device)
    elif model_type == 'resnet':
        model = load_resnet_model(device)
        preds, probs = predict_with_dl_model(model, spectrograms, device)
    else:
        model, scaler = load_classical_model(model_type)
        preds, probs = predict_with_classical_model(model, scaler, features)

    # Step 4: Aggregate across segments (average probabilities)
    print("[4/4] Aggregating results...\n")

    avg_probs = np.mean(probs, axis=0)
    final_pred = np.argmax(avg_probs)

    # Sort by confidence for top-k display
    sorted_indices = np.argsort(avg_probs)[::-1]

    print(f"{'='*60}")
    print(f"  PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"\n   Predicted Class: {CLASS_NAMES[final_pred]}")
    print(f"     Confidence:     {avg_probs[final_pred]*100:.1f}%\n")

    print(f"  Top-{min(top_k, N_CLASSES)} predictions:")
    print(f"  {'-'*46}")
    for rank, idx in enumerate(sorted_indices[:top_k]):
        bar = '#' * int(avg_probs[idx] * 30)
        print(f"  {rank+1}. {CLASS_NAMES[idx]:30s} {avg_probs[idx]*100:5.1f}% {bar}")

    # Per-segment breakdown
    if len(segments) > 1:
        print(f"\n  Per-segment predictions:")
        print(f"  {'-'*46}")
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            print(f"  Segment {i+1}: {CLASS_NAMES[pred]:30s} ({prob[pred]*100:.1f}%)")

    print(f"\n{'='*60}\n")

    return {
        'predicted_class': int(final_pred),
        'class_name': CLASS_NAMES[final_pred],
        'confidence': float(avg_probs[final_pred]),
        'all_probabilities': {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(N_CLASSES)},
        'per_segment_predictions': [int(p) for p in preds],
        'per_segment_probabilities': [p.tolist() for p in probs],
    }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Predict vessel class from underwater audio using trained VesselVoice models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vessel Classes:
  0 (A) — Motorboats
  1 (B) — Mussel Boats
  2 (C) — Fishing Vessels
  3 (D) — Passengers / Ferries
  4 (E) — Ocean Liners / Tugboats

Examples:
  python predict.py sample.wav
  python predict.py sample.wav --model resnet
  python predict.py sample.wav --model svm --top_k 5
        """
    )

    parser.add_argument('audio_path', type=str, help='Path to a .wav audio file')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet', 'svm', 'knn', 'rf'],
                        help='Model to use for prediction (default: cnn)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to display (default: 3)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for deep learning inference (default: cpu)')

    args = parser.parse_args()

    # Auto-detect CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'

    result = predict(
        audio_path=args.audio_path,
        model_type=args.model,
        top_k=args.top_k,
        device=args.device
    )

    return result


if __name__ == '__main__':
    main()
