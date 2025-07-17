
# ✋🎯 SignVision – Hand Sign Recognition Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch&logoColor=white)  
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)  
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)  

> 🚀 **SignVision** predicts hand signs (A-Z, `space`, `del`, `nothing`) using a PyTorch deep learning model.  
This project demonstrates **computer vision skills**, modular **Python code**, and modern ML workflows.  

---

## 📚 Table of Contents
- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
    - [1️⃣ Clone the Repository](#1️⃣-clone-the-repository)
    - [2️⃣ Install Dependencies](#2️⃣-install-dependencies)
    - [3️⃣ Run Prediction](#3️⃣-run-prediction)
- [⚙️ Requirements](#️-requirements)
- [📜 License](#-license)
- [👤 Author](#-author)

---

## ✨ Features
✅ Predicts **29 hand signs**: A-Z, `space`, `del`, `nothing`  
✅ Includes a **reusable Python module** (`signvision_module.py`)    
✅ Built with **PyTorch**, **Pandas**, and **Pillow**  

---

## 📂 Project Structure
```
SignVision-Model/
│──Final Project - contains the model built from scratch , training , validation and testing of model on real life data with plots .
├── LICENSE
└── README.md
├── model
├── signvision_module.py
├── SignVision Report 
├── requirements.txt

```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Himanshu-738/SignVision-Model.git
cd SignVision-Model
```

---

### 2️⃣ Install Dependencies
Install Python packages listed in `requirements.txt`:  
```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Prediction
```python
from SignVision_module import *

# Initialize
model = SignVision('SignVision_model')

# Predict sign from image
sign, confidence = model.predict("test_images/test1.jpg")
print(f"Predicted Sign: {sign} ({confidence:.2f} confidence)")
```

---

## ⚙️ Requirements
```
numpy== 1.26.4
pandas== 2.2.3
matplotlib== 3.9.2
seaborn== 0.13.2
scikit-learn== 1.5.1
torch== 2.7.1
torchvision== 0.22.1
Pillow== 10.4.0
Python==3.12.7
```

Install using:
```bash
pip install -r requirements.txt
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Himanshu Yadav**  
🌐 GitHub: [Himanshu-738](https://github.com/Himanshu-738)  
📧 Email: himanshuyadav1961@gmail.com

