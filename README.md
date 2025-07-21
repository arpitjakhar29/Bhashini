# 🤟 Bhashini – ASL to Text Translator

Bhashini is a **real-time web application** that translates **American Sign Language (ASL)** into text, making communication more accessible for the hearing-impaired community. The system integrates computer vision and deep learning to detect hand gestures and convert them into grammatically correct English text.

---

## 📌 Key Features

- 🎥 **Real-time ASL Recognition** using webcam and MediaPipe
- 🤖 **Custom-trained Deep Learning Model** built with TensorFlow/Keras
- 🧠 **Grammar Correction** with LanguageTool API
- 🌐 **Fast & Responsive Frontend** using Vite + React
- 🧪 Achieved **98% accuracy** on test data
- 🔁 Smooth, low-latency translation experience

---

## 🛠️ Tech Stack

| Layer       | Technology                      |
|------------|----------------------------------|
| Frontend   | Vite + React.js                  |
| Backend    | Flask (Python)                   |
| Gesture Tracking | MediaPipe                  |
| Model Training | TensorFlow, Keras            |
| Grammar Check | LanguageTool API              |

---

## 📁 Project Structure

```
Bhashini/
├── client/                  # Vite + React frontend
│   ├── src/
│   ├── public/
│   └── ...
├── server/                  # Flask backend
│   ├── main.py
│   ├── model/               # Trained Keras model (.keras/.h5)
│   ├── utils/               # LanguageTool integration, preprocessing
│   └── ...
├── data/                    # Custom dataset used for training
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- Node.js 18+
- pip / virtualenv
- Webcam for real-time testing

### 🔨 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/arpitjakhar29/Bhashini.git
   cd Bhashini
   ```

2. **Start Backend (Flask)**
   ```bash
   cd server
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

3. **Start Frontend (Vite + React)**
   ```bash
   cd client
   npm install
   npm run dev
   ```

4. **Visit the app:**  
   Open `http://localhost:5173` in your browser.

---

## 🧪 Model Accuracy & Results

- **Training Data:** Custom dataset with labeled ASL gestures (A–Z, 0–9, and common signs)
- **Model Accuracy:** ~98% on validation set
- **Real-time FPS:** ~20–25 FPS on standard laptop with webcam
- **Improvement:** Integrated LanguageTool reduced grammar errors by ~80%

---



## 🙏 Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) for hand tracking
- [TensorFlow](https://www.tensorflow.org/) for model training
- [LanguageTool](https://languagetool.org/http-api/) for grammar correction

---

### 💡 Inspiration

This project aims to empower the hearing-impaired community by making ASL-to-text translation more accessible, accurate, and real-time—bridging communication gaps with technology.
