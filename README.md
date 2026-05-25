# ♻️ EcoScan — Garbage Detection System

An AI-powered Garbage Detection web app that classifies waste into 6 categories using **MobileNetV2** deep learning model.

---

## 🏫 Project Details
| Project Type | AI / Deep Learning — Image Classification |

---

## 📌 About the Project

EcoScan is an AI-based garbage detection system that uses a MobileNetV2 model trained on a dataset of garbage images. The user uploads an image and the system detects the type of waste, its danger level, and provides disposal tips.

---

## 🗑️ Garbage Classes Detected

| Class | Danger Level | Category |
|---|---|---|
| Cardboard | 🟢 LOW | Biodegradable / Recyclable |
| Glass | 🟡 MEDIUM | Non-Biodegradable / Recyclable |
| Metal | 🟡 MEDIUM | Non-Biodegradable / Recyclable |
| Paper | 🟢 LOW | Biodegradable / Recyclable |
| Plastic | 🔴 HIGH | Non-Biodegradable / Harmful |
| Trash | 🔴 HIGH | Non-Biodegradable / Mixed Waste |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Primary programming language |
| TensorFlow / Keras | Deep learning framework |
| MobileNetV2 | Pre-trained model (ImageNet weights) |
| Streamlit | Web app framework |
| NumPy | Numerical computations |
| Pillow (PIL) | Image processing |
| Scikit-learn | Model evaluation metrics |
| Matplotlib / Seaborn | Data visualization |
| OpenCV | Image and camera handling |
| GitHub | Code hosting and version control |
| Streamlit Cloud | Free web deployment platform |

---

## 📁 Project Structure

```
EcoScan/
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── ecoscan_best_model.h5     # Trained model (add after training)
├── README.md                 # Project documentation
└── dataset-resized/          # Dataset folder
    ├── TRAIN/
    │   ├── cardboard/
    │   ├── glass/
    │   ├── metal/
    │   ├── paper/
    │   ├── plastic/
    │   └── trash/
    └── TEST/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

---

## ✅ Features

- Upload any garbage image for instant classification
- Detects 6 types of waste: cardboard, glass, metal, paper, plastic, trash
- Shows confidence percentage for each class
- Displays danger level (LOW / MEDIUM / HIGH)
- Provides disposal tips for each waste type
- Clean dark-themed professional web UI

---

## ▶️ How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your trained model
Place `ecoscan_best_model.h5` in the same folder as `app.py`

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Upload all files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Click Deploy ✅

---

## 📄 License

This project was developed for educational purposes under **Edunet Foundation** guidance at **NSTI Mumbai-G**.
