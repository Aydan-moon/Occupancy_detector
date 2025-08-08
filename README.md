# 🏠 Occupancy Detection Dashboard

A **Streamlit-powered dashboard** for visualizing, predicting, and analyzing building occupancy using environmental sensor data.  
Supports **interactive visualizations**, **batch predictions**, and **single predictions** using a pre-trained model.

---

## 📌 Features

- **Upload CSV Data** or **use built-in sample data**  
- **Occupancy trends over time** (if `Date` column is present)  
- **Correlation heatmap** of numeric features  
- **Feature importance chart** (if supported by the model)  
- **Batch predictions** with downloadable results  
- **Model performance metrics** (Accuracy, Precision, Recall)  
- **Confusion matrix** & **ROC curve**  
- **Single prediction** using sliders with unit labels & tooltips  

---

## 📂 Project Structure

.
├── app.py # Streamlit application
├── models/
│ └── occupancy_final_model.pkl # Pre-trained model (required)
├── Occupancy.ipynb # Jupyter notebook with model training/analysis
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/occupancy-dashboard.git
   cd occupancy-dashboard
Create a virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Ensure model file is available
Place your trained model in:

bash
Copy
Edit
models/occupancy_final_model.pkl
Run the app

bash
Copy
Edit
streamlit run app.py
Open in your browser

arduino
Copy
Edit
http://localhost:8501
📊 Example CSV Format
Your CSV should contain sensor readings and optionally a Date and Occupancy column:

Date	Temperature	Light	CO2	HumidityRatio	Occupancy
2025-08-01	21.5	350.5	420	0.0045	1
2025-08-02	22.0	340.2	410	0.0044	0

🖥 How to Use
Upload your CSV from the sidebar or click "Use Sample Data" to try the dashboard instantly.

Explore:

Occupancy trends over time

Sensor data correlations

Feature importance (if available)

Use Batch Prediction to predict occupancy for your dataset and download results.

Use Single Prediction in the sidebar by adjusting sliders for:

Temperature (°C)

Light (lux)

CO₂ (ppm)

Humidity Ratio

📦 Requirements
Python 3.8+

Streamlit

Pandas

NumPy

Seaborn

Matplotlib

Plotly

scikit-learn

joblib

Install them all:

bash
Copy
Edit
pip install -r requirements.txt
📸 Screenshots
(Replace these with actual screenshots after running the app)

Main Dashboard View

Single Prediction Sidebar

✨ Author
Developed by Aydan
📧 Contact: aydanrzyv@gmail.com