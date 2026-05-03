================================================================================
  UNDERWATER VESSEL NOISE CLASSIFICATION USING SHIPSEAR DATASET
  PVR Lab Course Project — Batch 5
================================================================================

AUTHORS
-------
  Somya Pratap Singh   | SAP ID: 500124413
  Akshat Choudhary     | SAP ID: 500124119
  Kapil                | SAP ID: 500119132

  School of Computer Science
  University of Petroleum & Energy Studies (UPES), Dehradun, India

================================================================================
DATASET
================================================================================

Dataset Name : ShipsEar — shipsear_5s_16k
Dataset Link : https://drive.google.com/drive/folders/1PrO_ZBiYM9hmMQIdgqt7otH0886pO7Xr

Description  : 90 underwater hydrophone recordings across 5 vessel classes:
               Class 0 (A) — Motorboats
               Class 1 (B) — Mussel Boats
               Class 2 (C) — Fishing Vessels
               Class 3 (D) — Passengers / Ferries
               Class 4 (E) — Ocean Liners / Tugboats

               Pre-segmented into 5-second clips at 16 kHz.
               Provided with train_list.txt / test_list.txt split files.

================================================================================
PROJECT STRUCTURE
================================================================================

├───Part 1 Background and motivation
├───Part 2 data
│   ├───Data acquisition and exploration
│   └───Exploratory data analysis
│       └───outputs_reports
├───Part 3 Preprocessing
│   ├───Resampling and normalization
│   │   ├───output_reports
│   │   └───shipsear_processed
│   │       ├───0
│   │       │   ├───0_0
│   │       │   ├───0_1
│   │       │   ├───0_2
│   │       │   ├───0_3
│   │       │   └───0_4
│   │       ├───1
│   │       │   ├───1_0
│   │       │   ├───1_1
│   │       │   └───1_2
│   │       ├───2
│   │       │   └───2_0
│   │       ├───3
│   │       │   ├───3_0
│   │       │   └───3_1
│   │       └───4
│   │           └───4_0
│   └───Segmentation and data splitting
├───Part 4  Feature extraction
│   ├───Deep feature representation
│   │   └───spectogram data tensors
│   └───Hand crafted features
├───Part 5 Classification models
│   ├───Baseline classical classifier models
│   │   └───models
│   └───Deep learning  CNN on mel log spectrogram (MFCC)
│       └───model weights
└───Part 6 Performance Evaluation
    ├───Ablation study and error analysis
    └───Quantitative Evaluation on test data
|--Shipsear_report.txt
|--Shipsear_report.tex
|-- README.txt                                                      

================================================================================
ENVIRONMENT & DEPENDENCIES
================================================================================

Platform : Google Colaboratory (recommended)
GPU      : NVIDIA Tesla T4 (select in Runtime > Change runtime type)
Python   : 3.12

--- Install all dependencies in Colab by running this cell first ---

  !pip install librosa soundfile resampy soxr tqdm Pillow \
               scikit-learn numpy pandas matplotlib seaborn scipy \
               torch torchvision

  NOTE: Do NOT run "pip install torch" separately on Colab —
        this overwrites the pre-installed GPU-enabled version.
        The above command only installs missing packages.

--- Full dependency list ---

  Core Audio:
    librosa       >= 0.10.0
    soundfile     >= 0.12.0
    resampy       >= 0.4.2     (required for librosa kaiser_best resampling)
    soxr          >= 0.3.7     (alternative resampler)

  Machine Learning:
    scikit-learn  >= 1.4.0
    torch         >= 2.0.0     (with CUDA support on Colab GPU)
    torchvision   >= 0.15.0

  Data & Visualisation:
    numpy         >= 1.26.0
    pandas        >= 2.0.0
    matplotlib    >= 3.7.0
    seaborn       >= 0.13.0
    scipy         >= 1.11.0
    Pillow        >= 10.0.0
    tqdm          >= 4.65.0

================================================================================
HOW TO RUN
================================================================================

STEP 0 — Setup
--------------
1. Open Google Colab: https://colab.research.google.com
2. Set runtime to GPU:
   Runtime > Change runtime type > Hardware accelerator > T4 GPU > Save
3. Upload all notebooks OR mount Google Drive:

   from google.colab import drive
   drive.mount('/content/drive')

4. Download dataset from:
   https://drive.google.com/drive/folders/1PrO_ZBiYM9hmMQIdgqt7otH0886pO7Xr
   and place in your Drive or Colab /content/ directory.

5. Install dependencies:
   !pip install librosa soundfile resampy soxr tqdm Pillow scikit-learn


STEP 1 — Part 2: EDA
---------------------
  Notebook : shipsEar_2.ipynb
  Action   : Update DATASET_ROOT to your local path of shipsear_5s_16k/
  Run all  : Runtime > Run all
  Outputs  : dataset_manifest.csv + 8 figure PNGs


STEP 2 — Part 3: Preprocessing & Splits
-----------------------------------------
  Notebook : shipsEar_2.ipynb
  Action   : Update DATASET_ROOT and PROCESSED_ROOT paths
  Run all  : Runtime > Run all
  Outputs  : shipsear_processed/ folder + splits.json


STEP 3 — Part 4: Feature Extraction
-------------------------------------
  Notebook : Part4_Feature_Extraction.ipynb
  Action   : Update DATASET_ROOT path in Cell 0
             If splits.json paths are from a different machine, run
             the path remapping cell (Section 1 Fix) first
  Run all  : Runtime > Run all (takes ~20-40 min on CPU, ~5 min on GPU)
  Outputs  : features/ folder with .npy arrays and scaler.pkl


STEP 4 — Part 5: Classification Models
----------------------------------------
  Notebook : ShipsEar_5.ipynb
  Action   : Run cells in order:
             - Cell 0: Config (verify DEVICE shows 'cuda')
             - Cell 1: Load features
             - Cells 2-6: KNN, SVM, Random Forest
             - Cells 7-12: CNN and ResNet-18 training
             - Cells 13-16: GradCAM, confusion matrices, ROC, results
  Note     : CNN training ~6 min on T4, ResNet-18 ~15 min on T4
  Outputs  : models/*.pkl  models/*.pth  + figures


STEP 5 — Part 6: Evaluation & Ablation
----------------------------------------
  Notebook : Part6_Evaluation.ipynb
  Action   : Run cells in order
             Cell 1b (resplit fix) must be run if class 3 is missing
             from training (check diagnostic output of Cell 1)
  Outputs  : final_results_table.csv, ablation_results.csv,
             literature_comparison.csv + 6 figures

================================================================================
KNOWN ISSUES & FIXES
================================================================================

Issue 1: ModuleNotFoundError: No module named 'resampy'
  Fix   : !pip install resampy soxr
          OR change res_type='kaiser_best' to res_type='soxr_hq'

Issue 2: TypeError: ReduceLROnPlateau got unexpected keyword 'verbose'
  Fix   : Remove verbose=False from ReduceLROnPlateau call
          (removed in PyTorch >= 2.4)

Issue 3: splits.json has 0 paths / Train = 0 segments
  Fix   : Run Part3_Splits_Fix_v2.ipynb
          The list files use absolute paths from original machine (E:\MTQP\...)
          The fix remaps them to your local DATASET_ROOT automatically.

Issue 4: val_loss = nan during CNN training
  Fix   : Run the val-set resplit cell (pool train+val, stratified split)
          Cause: all segments of one class ended up in val only.

Issue 5: IndexError: y_prob axis 1 out of bounds (size 4 not 5)
  Fix   : Run pad_proba_missing_class3() cell
          Cause: one class absent from training → sklearn outputs 4 columns.

Issue 6: CNN/ResNet-18 low accuracy (0.25-0.33) on base split
  Reason: Class 3 (Passengers/Ferries) was missing from training set.
  Fix   : Use 5-seed mean results which use proper stratified splits.
          5-seed mean: CNN=0.825, ResNet-18=0.902.

================================================================================
HARDWARE CONFIGURATION USED
================================================================================

  Platform  : Google Colaboratory (Free Tier)
  GPU       : NVIDIA Tesla T4 (16 GB VRAM)
  CPU       : Intel Xeon (2 vCPU)
  RAM       : 12 GB
  PyTorch   : 2.10.0+cu128
  CUDA      : 12.8

================================================================================
REFERENCES
================================================================================

[1] Santos-Dominguez et al., "ShipsEar: An underwater vessel noise database,"
    Applied Acoustics, vol. 113, pp. 64-69, 2016.
    https://doi.org/10.1016/j.apacoust.2016.06.008

[2] Irfan et al., "DeepShip: An underwater acoustic benchmark dataset,"
    Expert Systems with Applications, vol. 183, 2021.

[3] Xie et al., "Underwater target recognition using CNN with data
    augmentation," Applied Acoustics, vol. 193, 2022.

[4] He et al., "Deep residual learning for image recognition," CVPR, 2016.

[5] Park et al., "SpecAugment," Interspeech, 2019.

================================================================================