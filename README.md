# 🚗 Car Logo & Number Plate Detection

This project uses **TensorFlow Object Detection API** and **Xception CNN** to detect cars, recognize their brand logos, and read number plates from images or videos.  

---

## 📌 Project Overview

- **Object Detection:**  
  Trained with TensorFlow’s **ResNet 640×640** architecture.  
  Dataset was **self-collected** and annotated using **LabelImg** for:
  - Car Logos
  - Number Plates  

- **Logo Brand Classification:**  
  Used **Xception CNN architecture** (an advanced version of Inception V3).  
  Trained to predict one of 8 brands:  
  > Hyundai, Lexus, Mercedes, Opel, Skoda, Toyota, Volkswagen, Mazda  
  Achieved **86% validation accuracy**.

- **Number Plate Recognition:**  
  Used **EasyOCR** to read and produce text from number plates.

- **Deployment:**  
  Integrated with **Gradio Interface** for quick detection demos.

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>





