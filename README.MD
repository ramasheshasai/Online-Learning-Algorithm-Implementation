# 📘 Online Learning Algorithm for Real-Time Logistic Regression

## 📌 Project Overview
This project implements a **real-time Online Learning System** using **Online Logistic Regression** with **Stochastic Gradient Descent (SGD)**. Instead of training once on a fixed dataset like traditional machine learning, this project continuously learns from incoming streaming data and updates the model after every data instance.

This approach is essential for real-world systems where data arrives continuously such as:
- Fraud Detection Systems
- Stock Market Streaming Data
- IoT Sensor Data
- Clickstream Behavior
- Healthcare Monitoring
- Network Intrusion Detection

Traditional ML models fail in such environments because:
- They require full dataset before training  
- They cannot adapt to new behavior  
- Retraining frequently is expensive  

This project solves that problem using **Online Learning**.

---

## 🎯 What This Project Does
This project builds a complete real-time learning pipeline with:

### ✅ Online Logistic Regression Model
- Learns continuously instead of once
- Updates weights using SGD after every sample
- Uses Learning Rate Scheduling for stability
- Supports adaptive training

---

### ✅ Streaming Data Engine
- Reads dataset row-by-row
- Simulates real-time data streaming
- Can add artificial delay (like real streaming environments)

So instead of:
```
Train once → Stop
```
We do:
```
Receive data → Predict → Learn → Continue Forever
```

---

### ✅ Sliding Window Evaluation
Instead of evaluating on entire dataset, system evaluates based on **recent performance only** using a sliding window.

Continuously calculates:
- Accuracy
- Precision
- Recall
- F1 Score

This replicates real monitoring where:
✔ Recent performance is more important  
✔ Old history should not dominate

---

### ✅ Concept Drift Detection
In real systems, data patterns change over time (called **Concept Drift**).  
Example:
- Fraudsters change tactics
- User behavior evolves
- Market patterns shift

This system:
- Monitors recent prediction errors
- Detects when performance drops
- Automatically resets / adapts model
- Makes system self-learning and robust

---

### ✅ Real-Time Live Graph Dashboard
The project has a **live updating accuracy graph** showing:
- Performance over time
- Stability trends
- Recovery after drift

Perfect for:
- Demonstrations
- Monitoring
- Viva & presentations

---

### ✅ Batch Logistic Regression Comparison
To prove correctness, a Jupyter Notebook:
- Trains normal Logistic Regression on batch data
- Computes Accuracy & F1 Score
- Generates:
  - Confusion Matrix
  - ROC Curve
  - Precision–Recall Curve

This shows:
✔ Online model performs well  
✔ Comparable to batch model  
✔ Scientifically validated

---

## 🏗 Project Structure
```
online-learning/
│
├── src/
│   ├── online_logistic.py        # Online Logistic Regression Model
│   ├── metrics.py                # Sliding Window Metrics
│   ├── streaming.py              # Streaming Simulation
│   ├── drift.py                  # Concept Drift Detection
│   ├── visualizer.py             # Real-Time Accuracy Graph
│
├── notebooks/
│   ├── evaluation.ipynb          # Batch vs Online Comparison
│
├── data/
│   ├── stream.csv                # Streaming Dataset
│
├── generate_data.py              # Big Dataset Generator
├── main.py                       # Main Streaming Engine
└── README.md
```

---

## ⚙️ Installation
Install required dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 🗂 Dataset
Generate dataset:
```bash
python generate_data.py
```

Dataset stored at:
```
data/stream.csv
```

Format:
```
feat1,feat2,feat3,...,target
```
Target = 0 or 1 (binary classification)

---

## ▶️ Run Real-Time Streaming Model
Run:
```bash
python main.py
```

You will see:
- Continuous streaming updates
- Sliding window metrics
- Drift detection alerts
- Live accuracy graph

Example Output:
```
Seen 1000 | {'accuracy': 0.85, 'precision': 0.86, 'recall': 0.84, 'f1': 0.85}
Seen 2000 | ...
⚠️ Concept Drift Detected — Resetting Model
```

---

## 🧪 Batch vs Online Evaluation
Run:
```bash
jupyter notebook
```
Open:
```
notebooks/evaluation.ipynb
```

You get:
- Batch Accuracy
- Batch F1 Score
- Confusion Matrix
- ROC Curve
- Precision–Recall Graph

Example:
```
Batch Logistic Regression Accuracy: 0.9995
Batch Logistic Regression F1: 0.9995
```

---

## 🧠 Mathematical Idea (Simple)
Logistic Regression predicts:
```
p = 1 / (1 + e^(-wx + b))
```

Model Update (SGD):
```
error = predicted − actual
weights = weights − learning_rate * error * x
bias = bias − learning_rate * error
```

Learning Rate Decay:
```
lr = initial_lr / sqrt(iteration)
```

Sliding Window:
```
Evaluate last N predictions only
```

Drift:
```
If recent error > threshold → reset model
```

---

## 🧠 Real-World Applications
This approach is widely used in:
- Banking Fraud Detection
- Real-Time Recommendation Systems
- Behavior Monitoring
- Cyber Security
- Medical Streaming Systems
- Stock Trading Platforms
- Live AI Systems

---

## 🚀 Future Enhancements
- Real streaming (Kafka, MQTT, WebSockets)
- Advanced drift detection (ADWIN, DDM)
- Online SVM / Neural Networks
- Model Checkpoint Saving & Loading
- Web Dashboard UI
- Deployment as API

---

## ✅ Final Outcome
This project is:
✔ Research-grade  
✔ Industry relevant  
✔ Highly practical  
✔ Viva & submission ready  
✔ Resume strong  
✔ Technically impressive  

A complete **Online Machine Learning System**, not just a normal ML program.

---
