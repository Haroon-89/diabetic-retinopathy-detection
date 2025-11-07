# 🩺 Diabetic Retinopathy Detection System

An AI-powered web application for automated detection of Diabetic Retinopathy (DR) from retinal fundus images using deep learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

## 🔍 Overview

Diabetic Retinopathy is a diabetes complication that affects the eyes, potentially leading to blindness if left untreated. This application uses a deep learning model based on EfficientNetB3 architecture to automatically detect signs of diabetic retinopathy from retinal images.

**Key Statistics:**
- **Model Accuracy:** 96.97%
- **AUC Score:** 0.995
- **Architecture:** EfficientNetB3
- **Training Time:** ~14 minutes

## ✨ Features

### Core Functionality
- 🖼️ **Single Image Analysis**: Upload and analyze individual retinal images
- 📊 **Batch Processing**: Process multiple images simultaneously
- 📈 **Confidence Scores**: View prediction confidence with visual progress bars
- 📄 **Downloadable Reports**: Generate detailed text reports for each analysis
- 💾 **CSV Export**: Download batch analysis results in CSV format

### User Interface
- 🎨 Modern, responsive design with gradient styling
- 📱 Mobile-friendly layout
- 🔄 Real-time progress indicators
- 📊 Interactive data visualization
- ℹ️ Educational information about Diabetic Retinopathy

### Technical Features
- ⚡ Fast inference using TensorFlow/Keras
- 🔄 Automatic image preprocessing
- 🎯 High accuracy predictions (96.97%)
- 📦 Efficient model caching
- 🛡️ Robust error handling

## 📁 Project Structure

```
diabetic-retinopathy-detection/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/
│   ├── best_model.h5              # Trained EfficientNetB3 model
│   ├── dr_model_final.h5          # Alternative model file
│   └── metadata.json              # Model metadata and statistics
│
├── sample_images/
│   ├── dr1.jpg                    # Sample DR positive image
│   ├── dr2.jpg                    # Sample DR positive image
│   ├── dr3.jpg                    # Sample DR positive image
│   ├── dr4.jpg                    # Sample DR positive image
│   ├── dr5.jpg                    # Sample DR positive image
│   ├── n_dr.jpg                   # Sample no DR image
│   ├── n_dr2.jpg                  # Sample no DR image
│   └── n_dr3.jpg                  # Sample no DR image
│
├── favicon/
│   └── logo.png                   # Favicon for web app
│
└── .venv/                         # Virtual environment (optional)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Haroon-89/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model Files

Download the trained model files and place them in the `models/` directory:

1. **best_model.h5** or **dr_model_final.h5** - The trained model
2. **metadata.json** - Model metadata

### Step 5: Verify Installation with Test Images

The repository includes sample test images for verification:
- **DR positive images**: `dr1.jpg` to `dr5.jpg`
- **No DR images**: `n_dr.jpg`, `n_dr2.jpg`, `n_dr3.jpg`

Use these images to test the application functionality.

## 🎮 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Single Image Analysis

1. Click on "Choose a retinal fundus image"
2. Select a retinal image (JPG, JPEG, or PNG)
3. Click "Analyze Image" button
4. View the prediction results and confidence score
5. Download the detailed report (optional)

### Batch Processing

1. Expand the "Batch Analysis (Multiple Images)" section
2. Upload multiple retinal images
3. Click "Process All Images"
4. View results in a table format
5. Download CSV report with all results

### Testing with Sample Images

The application includes test images in the `sample_images/` folder:

**Diabetic Retinopathy Positive Images:**
- `dr1.jpg` - `dr5.jpg`: Images with signs of diabetic retinopathy

**No Diabetic Retinopathy Images:**
- `n_dr.jpg`, `n_dr2.jpg`, `n_dr3.jpg`: Healthy retinal images

**Quick Test:**
```bash
# After starting the app, try these tests:
1. Upload sample_images/dr1.jpg → Should detect DR
2. Upload sample_images/n_dr.jpg → Should show No DR
3. Batch process all 8 sample images → Should show mixed results
```

## 🧠 Model Information

### Architecture
- **Base Model:** EfficientNetB3
- **Framework:** TensorFlow/Keras 2.15
- **Input Shape:** 224 × 224 × 3 (RGB)
- **Output Classes:** 2 (DR, No_DR)

### Training Details
- **Dataset:** Diabetic Retinopathy Detection Dataset
- **Epochs:** 25
- **Batch Size:** 32
- **Learning Rate:** 0.0001
- **Dropout:** 0.3

### Performance Metrics
```json
{
  "test_accuracy": 0.9697,
  "test_auc": 0.9955,
  "training_time_minutes": 13.6
}
```

### Preprocessing Pipeline
1. Image resizing to 224×224 pixels
2. RGB conversion (if needed)
3. EfficientNet-specific preprocessing
4. Normalization

## 📸 Screenshots

### Main Interface
![Main Interface](screenshots/main_interface.png)

### Positive Detection
![DR Detected](screenshots/dr_detected.png)

### Negative Detection
![No DR](screenshots/no_dr.png)

### Batch Processing
![Batch Processing](screenshots/batch_processing.png)

## ⚠️ Disclaimer

**IMPORTANT: FOR EDUCATIONAL PURPOSES ONLY**

This application is designed for educational and research purposes only. It is **NOT** a medical device and should **NOT** be used for:

- Clinical diagnosis
- Treatment decisions
- Patient care without professional medical oversight
- Replacement of professional medical advice

**Always consult qualified healthcare professionals for:**
- Medical diagnosis
- Treatment recommendations
- Professional eye examinations
- Clinical decision-making

## 🔧 Troubleshooting

### Model Not Loading

**Error:** "Model file not found"

**Solution:**
1. Ensure model files are in the `models/` directory
2. Check file names match: `best_model.h5` or `dr_model_final.h5`
3. Verify file permissions

### Import Errors

**Error:** "ModuleNotFoundError"

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Image Upload Issues

**Error:** Image fails to process

**Solution:**
- Ensure image is in JPG, JPEG, or PNG format
- Check image is not corrupted
- Verify image is a valid retinal fundus image
- Try with provided sample images first to verify setup

### Testing the Application

To verify everything is working correctly:

```bash
# 1. Start the application
streamlit run app.py

# 2. Test with provided samples
# Upload: sample_images/dr1.jpg → Expected: DR Detected
# Upload: sample_images/n_dr.jpg → Expected: No DR

# 3. Test batch processing
# Upload all 8 sample images → Expected: 5 DR, 3 No DR
```

## 📦 Requirements

```txt
streamlit>=1.28.0
tensorflow>=2.20.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

## 📝 License

This project is licensed for **Educational Use Only**. 

**Restrictions:**
- Not for commercial use
- Not for clinical/medical use
- Not for diagnosis or treatment

## 👥 Authors

- **Haroon Iqbal** - *Initial work* - [YourGitHub](https://github.com/Haroon-89)

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- TensorFlow and Keras teams
- Streamlit for the amazing framework
- Diabetic Retinopathy Detection Dataset contributors
- Medical professionals for domain expertise

## 📞 Contact

For questions, suggestions, or issues:

- **Email:** harooniqbal829@gmail.com
- **GitHub Issues:** [Create an issue](https://github.com/Haroon-89/diabetic-retinopathy-detection/issues)
- **LinkedIn:** [Your LinkedIn](https://linkedin.com/in/haroon-iqbal-drabu)

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Diabetic Retinopathy Information](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy)

## 📊 Roadmap

### Planned Features
- [ ] Multi-class classification (severity levels)
- [ ] Grad-CAM visualization
- [ ] User authentication system
- [ ] Database integration for history
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Model explainability dashboard

### Future Improvements
- [ ] Support for additional retinal conditions
- [ ] Real-time camera integration
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Cloud deployment guide

---

**⭐ If you find this project helpful, please consider giving it a star!**

**Made with ❤️ for the AI and Healthcare community**
