from flask import Flask, render_template, request, redirect, url_for, flash, session
from model import CarMaintenancePredictor
import pandas as pd
from datetime import datetime, timedelta
import os
import uuid
import json
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generates a secure random 32-character hex string

# Initialize search history list
search_history = []

# Initialize predictor and load data
predictor = CarMaintenancePredictor()

# Check if model exists, otherwise train a new one
if os.path.exists('car_maintenance_model.pkl'):
    predictor.load_model('car_maintenance_model.pkl')
    print("Loaded existing model")
else:
    # Generate synthetic data
    print("Generating synthetic data...")
    vehicles_df, services_df, fuel_logs_df, issues_df = predictor.generate_synthetic_data()
    
    # Preprocess data
    print("Preprocessing data...")
    features_df = predictor.preprocess_data(vehicles_df, services_df, fuel_logs_df, issues_df)
    
    # Train model
    print("Training model...")
    predictor.train_model(features_df)
    
    # Save model
    predictor.save_model("car_maintenance_model.pkl")
    print("Model saved as car_maintenance_model.pkl")

# Store data in memory for demo purposes
# In a real application, this would be a database
vehicles_df, services_df, fuel_logs_df, issues_df = predictor.generate_synthetic_data()

# Convert DataFrames to dictionaries for easier access
vehicles = vehicles_df.to_dict('records')
services = services_df.to_dict('records')
fuel_logs = fuel_logs_df.to_dict('records') if not fuel_logs_df.empty else []
issues = issues_df.to_dict('records') if not issues_df.empty else []

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    # Get counts for dashboard
    vehicle_count = len(vehicles)
    service_count = len(services)
    maintenance_needed_count = sum(1 for v in vehicles if predict_maintenance_needed(v['vehicle_id']))
    
    # Get recent services
    recent_services = sorted(services, key=lambda x: x['service_date'], reverse=True)[:5]
    
    # Get vehicles that need maintenance
    vehicles_needing_maintenance = []
    for vehicle in vehicles:
        if predict_maintenance_needed(vehicle['vehicle_id']):
            vehicle_with_prediction = vehicle.copy()
            vehicle_with_prediction['prediction'] = get_prediction(vehicle['vehicle_id'])
            vehicles_needing_maintenance.append(vehicle_with_prediction)
    
    # Limit to top 3
    vehicles_needing_maintenance = vehicles_needing_maintenance[:3]
    
    return render_template('index.html', 
                          vehicles=vehicles,
                          vehicle_count=vehicle_count,
                          service_count=service_count,
                          maintenance_needed_count=maintenance_needed_count,
                          recent_services=recent_services,
                          vehicles_needing_maintenance=vehicles_needing_maintenance)

@app.route('/vehicles')
def vehicle_list():
    return render_template('vehicles.html', vehicles=vehicles)

@app.route('/vehicle/<vehicle_id>')
def vehicle_details(vehicle_id):
    # Find the vehicle
    vehicle = next((v for v in vehicles if v['vehicle_id'] == vehicle_id), None)
    if not vehicle:
        flash('Vehicle not found', 'danger')
        return redirect(url_for('index'))
    
    # Get services for this vehicle
    vehicle_services = [s for s in services if s['vehicle_id'] == vehicle_id]
    vehicle_services.sort(key=lambda x: x['service_date'], reverse=True)
    
    # Get fuel logs for this vehicle
    vehicle_fuel_logs = [f for f in fuel_logs if f['vehicle_id'] == vehicle_id]
    vehicle_fuel_logs.sort(key=lambda x: x['fuel_date'], reverse=True)
    
    # Get issues for this vehicle
    vehicle_issues = [i for i in issues if i['vehicle_id'] == vehicle_id]
    vehicle_issues.sort(key=lambda x: x['issue_date'], reverse=True)
    
    # Get prediction
    prediction = get_prediction(vehicle_id)
    
    return render_template('vehicle_details.html', 
                          vehicle=vehicle,
                          services=vehicle_services,
                          fuel_logs=vehicle_fuel_logs,
                          issues=vehicle_issues,
                          prediction=prediction)

@app.route('/services')
# Look for any instances where url_for('service_history') is used and change them to url_for('service_list')
# This might be in your templates or in your route functions

# For example, if you have something like:
# return redirect(url_for('service_history'))
# Change it to:
# return redirect(url_for('service_list'))
def service_list():
    return render_template('services.html', services=services, vehicles=vehicles)

@app.route('/add_vehicle', methods=['GET', 'POST'])
def add_vehicle():
    if request.method == 'POST':
        # Generate a new VIN
        vehicle_id = predictor.generate_vin()
        
        # Create new vehicle
        new_vehicle = {
            'vehicle_id': vehicle_id,
            'make': request.form['make'],
            'model': request.form['model'],
            'year': int(request.form['year']),
            'current_mileage': int(request.form['current_mileage']),
            'fuel_type': request.form['fuel_type'],
            'engine_type': request.form['engine_type']
        }
        
        # Add to vehicles list
        vehicles.append(new_vehicle)
        
        flash(f'Vehicle {new_vehicle["make"]} {new_vehicle["model"]} added successfully', 'success')
        return redirect(url_for('vehicle_details', vehicle_id=vehicle_id))
    
    return render_template('add_vehicle.html')

@app.route('/add_service/<vehicle_id>', methods=['GET', 'POST'])
def add_service(vehicle_id):
    # Find the vehicle
    vehicle = next((v for v in vehicles if v['vehicle_id'] == vehicle_id), None)
    if not vehicle:
        flash('Vehicle not found', 'danger')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Generate service ID
        service_id = f"SRV{len(services) + 10001}"
        
        # Create new service
        new_service = {
            'service_id': service_id,
            'vehicle_id': vehicle_id,
            'service_type': request.form['service_type'],
            'service_date': request.form['service_date'],
            'service_mileage': int(request.form['service_mileage']),
            'service_cost': float(request.form['service_cost']),
            'next_service_mileage': int(request.form['service_mileage']) + 5000  # Simple estimate
        }
        
        # Add to services list
        services.append(new_service)
        
        # Update vehicle mileage if service mileage is higher
        if new_service['service_mileage'] > vehicle['current_mileage']:
            vehicle['current_mileage'] = new_service['service_mileage']
        
        flash('Service record added successfully', 'success')
        return redirect(url_for('vehicle_details', vehicle_id=vehicle_id))
    
    # Get services for this vehicle
    vehicle_services = [s for s in services if s['vehicle_id'] == vehicle_id]
    vehicle_services.sort(key=lambda x: x['service_date'], reverse=True)
    
    return render_template('add_service.html', 
                          vehicle=vehicle, 
                          services=vehicle_services,
                          maintenance_types=predictor.maintenance_types[:-1])  # Exclude 'none'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        vehicle_id = request.form['vehicle_id']
        make = request.form['make']
        model = request.form['model']
        service_id = request.form['service_id']
        
        try:
            # Check if vehicle exists
            vehicle = next((v for v in vehicles if v['vehicle_id'] == vehicle_id), None)
            if not vehicle:
                # Get a list of available vehicle IDs to help the user
                available_vehicles = [{'id': v['vehicle_id'], 'make': v['make'], 'model': v['model']} 
                                     for v in vehicles[:5]]  # Show first 5 for brevity
                flash(f'Vehicle with VIN {vehicle_id} not found. Available vehicles: ' + 
                      ', '.join([f"{v['id']} ({v['make']} {v['model']})" for v in available_vehicles]), 'warning')
                return redirect(url_for('predict'))
                
            # Check if service exists
            service = next((s for s in services if s['service_id'] == service_id), None)
            if not service:
                # Get a list of available service IDs for this vehicle
                available_services = [s['service_id'] for s in services 
                                     if s['vehicle_id'] == vehicle_id][:5]  # Show first 5
                if available_services:
                    flash(f'Service ID {service_id} not found. Available service IDs for this vehicle: ' + 
                          ', '.join(available_services), 'warning')
                else:
                    flash(f'No services found for vehicle {vehicle_id}', 'warning')
                return redirect(url_for('predict'))
            
            # Convert DataFrames for prediction
            vehicles_df = pd.DataFrame(vehicles)
            services_df = pd.DataFrame(services)
            fuel_logs_df = pd.DataFrame(fuel_logs) if fuel_logs else pd.DataFrame()
            issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()
            
            result = predictor.predict_from_ids(
                vehicle_id, make, model, service_id,
                vehicles_df, services_df, fuel_logs_df, issues_df
            )
            
            # Save search history if user is logged in
            if 'user_id' in session:
                search_id = str(uuid.uuid4())
                search_record = {
                    'search_id': search_id,
                    'user_id': session['user_id'],
                    'vehicle_id': vehicle_id,
                    'make': make,
                    'model': model,
                    'service_id': service_id,
                    'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction_result': result
                }
                search_history.append(search_record)
            
            return render_template('prediction_result.html', result=result)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')
            return redirect(url_for('predict'))
    
    # Get sample vehicle and service for demonstration
    sample_vehicle = vehicles[0]
    sample_service = next((s for s in services if s['vehicle_id'] == sample_vehicle['vehicle_id']), None)
    
    # Get a few sample vehicles and their services to display to the user
    sample_data = []
    for v in vehicles[:5]:  # Get first 5 vehicles
        v_services = [s for s in services if s['vehicle_id'] == v['vehicle_id']]
        if v_services:
            sample_data.append({
                'vehicle_id': v['vehicle_id'],
                'make': v['make'],
                'model': v['model'],
                'service_id': v_services[0]['service_id']
            })
    
    return render_template('predict.html', 
                          vehicles=vehicles, 
                          services=services,
                          sample_vehicle=sample_vehicle,
                          sample_service=sample_service,
                          sample_data=sample_data)

# Helper functions
def predict_maintenance_needed(vehicle_id):
    """Quick check if maintenance is needed for dashboard"""
    # Find the vehicle
    vehicle = next((v for v in vehicles if v['vehicle_id'] == vehicle_id), None)
    if not vehicle:
        return False
    
    # Find the most recent service
    vehicle_services = [s for s in services if s['vehicle_id'] == vehicle_id]
    if not vehicle_services:
        return True  # No services means maintenance is needed
    
    # Sort by date (newest first)
    vehicle_services.sort(key=lambda x: x['service_date'], reverse=True)
    last_service = vehicle_services[0]
    
    # Simple rule-based logic for quick check
    days_since_last_service = (datetime.now() - datetime.strptime(last_service['service_date'], '%Y-%m-%d')).days
    mileage_since_last_service = vehicle['current_mileage'] - last_service['service_mileage']
    
    # If it's been more than 6 months or 7500 miles, maintenance is needed
    return days_since_last_service > 180 or mileage_since_last_service > 7500

def get_prediction(vehicle_id):
    """Get full prediction for a vehicle"""
    # Find the vehicle
    vehicle = next((v for v in vehicles if v['vehicle_id'] == vehicle_id), None)
    if not vehicle:
        return None
    
    # Find the most recent service
    vehicle_services = [s for s in services if s['vehicle_id'] == vehicle_id]
    if not vehicle_services:
        return None  # Can't predict without service history
    
    # Sort by date (newest first)
    vehicle_services.sort(key=lambda x: x['service_date'], reverse=True)
    last_service = vehicle_services[0]
    
    try:
        # Convert DataFrames for prediction
        vehicles_df = pd.DataFrame(vehicles)
        services_df = pd.DataFrame(services)
        fuel_logs_df = pd.DataFrame(fuel_logs) if fuel_logs else pd.DataFrame()
        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()
        
        result = predictor.predict_from_ids(
            vehicle_id, vehicle['make'], vehicle['model'], last_service['service_id'],
            vehicles_df, services_df, fuel_logs_df, issues_df
        )
        
        return result
        
    except Exception as e:
        print(f"Error predicting for vehicle {vehicle_id}: {str(e)}")
        return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # In a real application, you would validate against a database
        # For now, we'll just redirect to the dashboard
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # In a real application, you would save to a database
        flash('Account created successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/google_login')
def google_login():
    # This would normally use OAuth with Google
    flash('Logged in with Google successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

if __name__ == '__main__':
    app.run(debug=True)