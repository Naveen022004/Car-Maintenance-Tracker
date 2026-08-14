# 🚗 Car Maintenance Tracker

**Car Maintenance Tracker** is a Flask-based web application that helps users manage vehicle information, service history, fuel records, maintenance issues, and maintenance schedules. The project also uses a **Machine Learning model based on Random Forest** to predict whether a vehicle requires maintenance and identify the possible type of maintenance needed.

## ✨ Features

* 🚘 Add and manage vehicle details
* 🔧 Record and track vehicle service history
* ⛽ Maintain fuel consumption and mileage records
* ⚠️ Track vehicle maintenance issues
* 🤖 Predict whether maintenance is required using Machine Learning
* 🛠️ Predict possible maintenance types such as oil change, tire rotation, brake service, battery check, and filter replacement
* 📊 Calculate maintenance-related features from vehicle and service history
* 🆔 Generate unique 17-character VINs for vehicles
* 🌐 Web-based interface using Flask

## 🤖 Machine Learning

The project uses **Random Forest Classifier** models to predict vehicle maintenance requirements.

The prediction is based on features such as:

* Vehicle make and model
* Vehicle year
* Current mileage
* Fuel type
* Engine type
* Days since last service
* Mileage since last service
* Average MPG
* Number of services in the last 6 months

The trained model is stored in `car_maintenance_model.pkl`.

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **HTML/CSS**
* **Random Forest Machine Learning**

## 📁 Project Structure

```text
Car-Maintainence-Tracer/
│
├── app.py
├── model.py
├── car_maintenance_model.pkl
├── requirements.txt
├── README.md
└── .gitattributes
```

## 🚀 How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Then open the local server URL shown in the terminal.

## 🎯 Project Objective

The main objective of this project is to provide a simple vehicle maintenance management system while using Machine Learning to help predict upcoming maintenance requirements based on vehicle usage and service history.

**Keywords:** `Python` `Flask` `Machine Learning` `Random Forest` `Pandas` `Scikit-learn` `Car Maintenance` `Vehicle Tracking`
