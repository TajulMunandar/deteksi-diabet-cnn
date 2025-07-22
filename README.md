# 👁️ Diabetic Retinopathy Detection using CNN

This project uses a Convolutional Neural Network (CNN) to detect **Diabetic Retinopathy** from retinal images. The model classifies the severity of the disease based on eye images provided in the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) competition on Kaggle.

---

## 📂 Dataset

- Source: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- Description: Retinal fundus images labeled on a scale from 0 to 4:
  - 0 - No DR
  - 1 - Mild
  - 2 - Moderate
  - 3 - Severe
  - 4 - Proliferative DR

*Note: The `dataset/` folder is ignored from Git using `.gitignore`.*

---

## 🧠 Model Architecture

- Model: Custom **CNN** built using TensorFlow/Keras
- Layers:
  - Convolutional + ReLU
  - MaxPooling
  - Dropout
  - Dense (Fully Connected)
  - Softmax (for multi-class classification)

You can modify the CNN in the training script to improve performance.

---

## 🏗️ Project Structure


---

## ⚙️ Requirements

- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

Install with:

```bash
pip install -r requirements.txt
