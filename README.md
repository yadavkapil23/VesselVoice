# 🚢 VesselVoice: Underwater Vessel Noise Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org/)

**VesselVoice** is a professional-grade full-stack application designed to classify vessels based on their underwater acoustic signatures. By leveraging the **ShipsEar** dataset, the system uses advanced signal processing and deep learning to identify five distinct categories of marine vessels with high precision.

---

## 🌟 Key Features

*   **Multi-Model Benchmarking:** Compare results across **Custom CNN**, **ResNet-18**, **KNN**, and **SVM** models.
*   **Advanced Acoustic Analysis:** Real-time extraction of 334 features, including MFCCs, Spectral Centroids, and LOFAR/DEMON analysis.
*   **Premium Web Dashboard:** A sleek, dark-themed **React** UI with drag-and-drop audio uploads and animated result visualizations.
*   **High-Performance Backend:** **FastAPI** server optimized for rapid model inference and JSON serialization.
*   **Signal Visualization:** Interactive probability distribution charts and acoustic signature analysis.

---

## 🏗️ System Architecture

The project is split into two primary components:

### 1. The Inference Engine (`predict.py` & `server.py`)
A robust Python-based backend that handles:
*   **Audio Preprocessing:** Resampling to 16kHz and segmenting into 5-second windows.
*   **Feature Extraction:** Generating log-mel spectrograms for Deep Learning and mathematical feature vectors for Classical ML.
*   **Inference:** Loading pre-trained weights and executing model passes.

### 2. The Modern Dashboard (`frontend/`)
A high-end React application built with:
*   **Vite:** For ultra-fast development and bundling.
*   **Tailwind CSS:** For a premium glassmorphism aesthetic.
*   **Recharts:** For beautiful, responsive data visualization.

---

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   NPM or Yarn

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install torch torchvision torchaudio librosa fastapi uvicorn python-multipart scikit-learn numpy scipy Pillow
   ```
2. Start the FastAPI server:
   ```bash
   python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install NPM packages:
   ```bash
   npm install
   ```
3. Launch the development server:
   ```bash
   npm run dev
   ```

---

## 🛠️ Usage

### Via Web UI (Recommended)
1. Open your browser to `http://localhost:3000` (or `3001`).
2. Select your target AI model in the **Configuration** panel.
3. Drag and drop a `.wav` file into the upload area.
4. Click **Classify Acoustic Signal** to see the deep analysis.

### Via Command Line
You can also run classification directly from the terminal:
```bash
python predict.py "path/to/audio.wav" --model resnet --device cpu
```

---

## 📊 Dataset & Research
This project is based on the **ShipsEar** dataset, a database of underwater vessel sounds recorded in the Atlantic Ocean. The research pipeline covers:
*   Data resampling and normalization.
*   Log-Mel Spectrogram optimization.
*   Deep Learning architecture fine-tuning.
*   Performance evaluation across diverse acoustic environments.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements
*   The creators of the **ShipsEar** dataset for providing the acoustic recordings.
*   The open-source communities behind PyTorch, FastAPI, and React.

---

**VesselVoice** — *Listening to the pulse of the ocean.* 🌊⚓
