
🏠 Occupancy Detection Dashboard
A Streamlit-powered web application for visualizing, predicting, and analyzing building occupancy using sensor data.
It supports batch predictions, single predictions, interactive plots, and performance evaluation of a pre-trained model.

📌 Features
Upload CSV Data to view and analyze occupancy trends

Interactive Time Series Charts for occupancy over time

Correlation Heatmap of sensor data

Feature Importance Visualization (if available from the model)

Batch Predictions with downloadable results

Model Performance Metrics (Accuracy, Precision, Recall)

Confusion Matrix and ROC Curve

Single Prediction using sidebar input fields

📂 Project Structure
bash
Copy
Edit
.
├── app.py                         # Streamlit application code
├── models/
│   └── occupancy_final_model.pkl  # Pre-trained model
├── Occupancy.ipynb                # Jupyter notebook with analysis/training
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
🚀 Installation & Setup
Clone this repository

bash
Copy
Edit
git clone https://github.com/yourusername/occupancy-dashboard.git
cd occupancy-dashboard
Create a virtual environment (optional but recommended)

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
Run the app

bash
Copy
Edit
streamlit run app.py
Access in your browser

arduino
Copy
Edit
http://localhost:8501
📊 Example CSV Format
Your uploaded CSV should contain sensor readings and optionally a Date and Occupancy column:

Date	Temperature	Light	CO2	HumidityRatio	Occupancy
2025-08-01	21.5	350.5	420	0.0045	1
2025-08-01	22.0	340.2	410	0.0044	0

📦 Model
The app uses a pre-trained model stored in:

bash
Copy
Edit
models/occupancy_final_model.pkl
Make sure this file is available in the correct directory before running the app.

🛠 Requirements
Python 3.8+

Streamlit

Pandas

NumPy

Seaborn

Matplotlib

Plotly

scikit-learn

joblib

✨ Author
Developed by Aydan
📧 Contact: [aydanrzyv@gmail.com]