# 🐋 Marine Mammal Sound Classification | Deep Learning for Bioacoustics  
*Advanced neural networks for cetacean vocalization analysis using 7-decade Watkins Database recordings*

![Python](https://img.shields.io/badge/Python-3.7-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4-orange) ![Librosa](https://img.shields.io/badge/Audio_Processing-Librosa_0.8-yellowgreen) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-red) ![Pandas](https://img.shields.io/badge/Data_Engineering-Pandas_1.1-lightgrey)

## 🌊 **Project Overview**
**Objective**: Develop state-of-the-art classifiers for 32 marine mammal species using historical recordings (1940s-present) from the Watkins Marine Mammal Sound Database, addressing key challenges in:
- **Temporal variability**: Audio durations from 0.05s to 1260s
- **Recording heterogeneity**: VCR, cassettes, digital formats
- **Geographic diversity**: Global sampling locations

**Key Achievements**:
✅ **83.2% accuracy** with CNN architecture (60x560 MFCC features)  
✅ **57x parameter reduction** via duration thresholding (<6.6min)  
✅ **Novel adaptive sampling** ensuring fixed 22,295pt vectors  
✅ **End-to-end pipeline**: Web scraping → MFCC extraction → Model training


## 📊 **Key Visualizations**
### **Spectrogram Analysis**  
```python
# Spectrogram generation code
y, sr = librosa.load(audio_path)
S = librosa.stft(y)
D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(D, sr=sr, hop_length=512,
                        x_axis='time', y_axis='log')
```
![Spectrogram](https://github.com/gacuervol/DeepLearning-Bioacoustics/blob/main/figures/spectogram.png)  
*Scientific Value:*  
- Identified species-specific acoustic fingerprints  
- Guided MFCC parameter selection (n_mfcc=60 optimal)  
- Revealed recording artifacts needing preprocessing  

### 🔍 **Data Pipeline**
```python
# Adaptive Sampling Algorithm
def adaptive_sr(y_target=22295, duration):
    return int(y_target / duration)  # Dynamic sample rate calculation

# MFCC Feature Extraction
mfccs = librosa.feature.mfcc(y=y, sr=adaptive_sr(duration), n_mfcc=60)
```

### 🤖 **Model Architectures**
**1. Dense Network (32 species)**
```python
Model: "Dense_Network"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Dense_1 (128 units)          (None, 128)               7808      
Dropout_1 (40%)              (None, 128)               0         
Dense_2 (128 units)          (None, 128)               16512     
Dense_3 (32 outputs)         (None, 32)                4128      
=================================================================
Total params: 44,960
```

**2. CNN Architecture (Best Performance)**
```python
Model: "CNN_Encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Conv2D (4 filters)           (None, 280, 30, 4)        40        
MaxPooling2D                 (None, 140, 15, 4)        0         
Conv2D (16 filters)          (None, 70, 8, 16)         592       
Conv2D (32 filters)          (None, 18, 2, 32)         4640      
Dense (64 units)             (None, 64)                20544     
=================================================================
Total params: 46,392
```

## 📊 **Performance Metrics**

### Model Comparison
| Model Type       | Accuracy | Val Loss | Parameters | Inference Time |
|------------------|----------|----------|------------|----------------|
| Dense Network    | 70.2%    | 1.03     | 44,960     | 4ms/sample     |
| **CNN**          | **83.2%**| 0.71     | 46,392     | 10ms/sample    |

### Classification Report (CNN)
```text
                      precision  recall  f1-score   support
Fraser's Dolphin       0.92      1.00      0.96        11
Risso's Dolphin        1.00      0.91      0.95        11
Sperm Whale           0.44      1.00      0.62         4
Striped Dolphin        0.90      0.90      0.90        10
Walrus                0.75      0.86      0.80         7
```

![Confusion Matrix](https://github.com/gacuervol/DeepLearning-Bioacoustics/blob/main/figures/confussion_matrix.png)  
*Confusion matrix showing strong performance on frequently observed species*

## 🚀 **Key Innovations**

1. **Temporal Normalization**
   - Implemented dynamic sample rate calculation: `sr = 22,295 / duration`
   - Enabled fixed-length input vectors despite varying audio lengths

2. **Computational Optimization**
   - Reduced model parameters from 7M to 46K via duration filtering
   - Achieved 83% accuracy with <50K parameters

3. **Data Augmentation**
   - Spectral noise reduction using `noisereduce` library
   - Time-frequency masking for class imbalance mitigation

## 📂 **Repository Structure**
```bash
├── Data/
│   ├── raw_audio/          # Original .wav files (1,697 samples)
│   ├── processed/          # Standardized MFCC features
├── Models/
│   ├── Cetaceos_conv32.h5  # Best-performing CNN
│   ├── Cetaceos_32dense.h5 # Baseline dense model
├── Notebooks/
│   ├── Web_Scraping.ipynb  # Watkins DB automation
│   ├── MFCC_Extraction.ipynb # Feature engineering
│   ├── CNN_Training.ipynb  # Model development
```

## 🌍 **Research Applications**
- **Species Conservation**: Track endangered populations through vocal signatures
- **Ocean Noise Monitoring**: Baseline for anthropogenic impact studies
- **Migration Pattern Analysis**: Temporal-spatial distribution modeling

## 🔗 **Connect**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Geospatial_Data_Scientist-0077B5?logo=linkedin)](https://www.linkedin.com/in/giovanny-alejandro-cuervo-londo%C3%B1o-b446ab23b/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publications-00CCBB?logo=researchgate)](https://www.researchgate.net/profile/Giovanny-Cuervo-Londono)  
[![Email](https://img.shields.io/badge/Email-giovanny.cuervo101%40alu.ulpgc.es-D14836?style=for-the-badge&logo=gmail)](mailto:giovanny.cuervo101@alu.ulpgc.es)  

> 🐬 **Open for Collaboration**:  
> - Open to collaborations  
> - Contact via LinkedIn for consulting  
