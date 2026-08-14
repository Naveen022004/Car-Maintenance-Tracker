import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class CarMaintenancePredictor:
    def __init__(self):
        self.model = None
        self.maintenance_type_model = None
        self.label_encoders = {}
        self.features = ['make', 'model', 'year', 'current_mileage', 'fuel_type', 
                        'engine_type', 'days_since_last_service', 'mileage_since_last_service',
                        'avg_mpg', 'service_count_6mo']
        self.target = 'needs_maintenance'
        self.maintenance_types = ['oil_change', 'tire_rotation', 'brake_service', 
                                 'battery_check', 'filter_replacement', 'none']
        
    def generate_vin(self):
        """Generate a 17-character alphanumeric VIN"""
        # VIN characters (excluding I, O, Q for readability)
        chars = '0123456789ABCDEFGHJKLMNPRSTUVWXYZ'
        return ''.join(random.choice(chars) for _ in range(17))
        
    def generate_synthetic_data(self, num_vehicles=100, num_services_per_vehicle=20):
        """Generate synthetic dataset for demonstration"""
        # Vehicle information
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Tesla']
        models = {
            'Toyota': ['Camry', 'Corolla', 'RAV4', 'Prius'],
            'Honda': ['Accord', 'Civic', 'CR-V', 'Pilot'],
            'Ford': ['F-150', 'Explorer', 'Mustang', 'Escape'],
            'Chevrolet': ['Silverado', 'Equinox', 'Malibu', 'Tahoe'],
            'BMW': ['3 Series', '5 Series', 'X5', 'X3'],
            'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE'],
            'Tesla': ['Model 3', 'Model S', 'Model X', 'Model Y']
        }
        fuel_types = ['Gasoline', 'Diesel', 'Hybrid', 'Electric']
        engine_types = ['V4', 'V6', 'V8', 'Turbocharged', 'Electric']
        
        vehicles = []
        services = []
        fuel_logs = []
        issues = []
        
        for i in range(num_vehicles):
            make = random.choice(makes)
            model = random.choice(models[make])
            year = random.randint(2010, 2023)
            current_mileage = random.randint(5000, 150000)
            fuel_type = random.choice(fuel_types)
            engine_type = random.choice(engine_types)
            
            vehicle_id = self.generate_vin()
            vehicles.append({
                'vehicle_id': vehicle_id,
                'make': make,
                'model': model,
                'year': year,
                'current_mileage': current_mileage,
                'fuel_type': fuel_type,
                'engine_type': engine_type
            })
            
            # Generate service history
            last_service_date = datetime.now() - timedelta(days=random.randint(0, 365))
            last_service_mileage = max(0, current_mileage - random.randint(1000, 10000))
            
            for j in range(num_services_per_vehicle):
                service_date = last_service_date - timedelta(days=random.randint(30, 180))
                service_mileage = max(0, last_service_mileage - random.randint(1000, 5000))
                service_type = random.choice(self.maintenance_types[:-1])  # Exclude 'none'
                service_cost = round(random.uniform(50, 500), 2)
                
                services.append({
                    'service_id': f"SRV{10000 + i*num_services_per_vehicle + j}",
                    'vehicle_id': vehicle_id,
                    'service_type': service_type,
                    'service_date': service_date.strftime('%Y-%m-%d'),
                    'service_mileage': service_mileage,
                    'service_cost': service_cost,
                    'next_service_mileage': service_mileage + random.randint(3000, 10000),
                })
                
                last_service_date = service_date
                last_service_mileage = service_mileage
            
            # Generate fuel logs for non-electric vehicles
            if fuel_type != 'Electric':
                for k in range(random.randint(5, 20)):
                    fuel_date = datetime.now() - timedelta(days=random.randint(0, 90))
                    fuel_volume = round(random.uniform(8, 20), 2)
                    fuel_mileage = max(0, current_mileage - random.randint(100, 500))
                    mpg = round(random.uniform(20, 40), 1)
                    
                    fuel_logs.append({
                        'fuel_log_id': f"FL{10000 + i*20 + k}",
                        'vehicle_id': vehicle_id,
                        'fuel_date': fuel_date.strftime('%Y-%m-%d'),
                        'fuel_mileage': fuel_mileage,
                        'mpg': mpg,
                    })
            
            # Generate occasional issues
            if random.random() < 0.3:  # 30% chance of having issues
                issue_date = datetime.now() - timedelta(days=random.randint(0, 180))
                issue_mileage = max(0, current_mileage - random.randint(100, 5000))
                issue_type = random.choice(['Engine', 'Battery', 'Transmission', 'Electrical', 'Suspension'])
                
                issues.append({
                    'issue_id': f"ISS{10000 + i*5 + len(issues)}",
                    'vehicle_id': vehicle_id,
                    'issue_type': issue_type,
                    'issue_date': issue_date.strftime('%Y-%m-%d'),
                    'issue_mileage': issue_mileage,
                })
        
        # Create DataFrames
        vehicles_df = pd.DataFrame(vehicles)
        services_df = pd.DataFrame(services)
        fuel_logs_df = pd.DataFrame(fuel_logs) if fuel_logs else pd.DataFrame()
        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()
        
        return vehicles_df, services_df, fuel_logs_df, issues_df
    
    def preprocess_data(self, vehicles_df, services_df, fuel_logs_df, issues_df):
        """Prepare data for ML model training"""
        features_list = []
        
        for _, vehicle in vehicles_df.iterrows():
            vehicle_id = vehicle['vehicle_id']
            
            # Get all services for this vehicle
            vehicle_services = services_df[services_df['vehicle_id'] == vehicle_id]
            
            # Calculate service-related features
            if not vehicle_services.empty:
                last_service = vehicle_services.iloc[0]
                days_since_last_service = (datetime.now() - pd.to_datetime(last_service['service_date'])).days
                mileage_since_last_service = vehicle['current_mileage'] - last_service['service_mileage']
                service_count_6mo = len(vehicle_services[pd.to_datetime(vehicle_services['service_date']) > (datetime.now() - timedelta(days=180))])
            else:
                days_since_last_service = 365  # Default value if no services
                mileage_since_last_service = vehicle['current_mileage']
                service_count_6mo = 0
            
            # Calculate fuel efficiency for non-electric vehicles
            if vehicle['fuel_type'] != 'Electric' and not fuel_logs_df.empty:
                vehicle_fuel_logs = fuel_logs_df[fuel_logs_df['vehicle_id'] == vehicle_id]
                avg_mpg = vehicle_fuel_logs['mpg'].mean() if not vehicle_fuel_logs.empty else 25.0
            else:
                avg_mpg = 0.0  # Electric vehicles
            
            # Determine if maintenance is needed (our target variable)
            needs_maintenance = False
            maintenance_type = 'none'
            
            # Rule-based logic for synthetic data
            if mileage_since_last_service > 7500:
                needs_maintenance = True
                maintenance_type = 'oil_change'
            elif days_since_last_service > 180:
                needs_maintenance = True
                maintenance_type = random.choice(['tire_rotation', 'battery_check'])
            elif not issues_df.empty and len(issues_df[issues_df['vehicle_id'] == vehicle_id]) > 2:
                needs_maintenance = True
                maintenance_type = 'brake_service'
            
            features_list.append({
                'vehicle_id': vehicle_id,
                'make': vehicle['make'],
                'model': vehicle['model'],
                'year': vehicle['year'],
                'current_mileage': vehicle['current_mileage'],
                'fuel_type': vehicle['fuel_type'],
                'engine_type': vehicle['engine_type'],
                'days_since_last_service': days_since_last_service,
                'mileage_since_last_service': mileage_since_last_service,
                'avg_mpg': avg_mpg,
                'service_count_6mo': service_count_6mo,
                'needs_maintenance': needs_maintenance,
                'maintenance_type': maintenance_type
            })
        
        features_df = pd.DataFrame(features_list)
        
        # Encode categorical variables
        categorical_cols = ['make', 'model', 'fuel_type', 'engine_type']
        for col in categorical_cols:
            le = LabelEncoder()
            features_df[col] = le.fit_transform(features_df[col])
            self.label_encoders[col] = le
        
        return features_df
    
    def train_model(self, features_df):
        """Train the maintenance prediction model"""
        X = features_df[self.features]
        y = features_df[self.target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))
        
        # Train maintenance type classifier
        maintenance_df = features_df[features_df['needs_maintenance'] == True]
        if not maintenance_df.empty:
            X_maint = maintenance_df[self.features]
            y_maint = maintenance_df['maintenance_type']
            
            self.maintenance_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.maintenance_type_model.fit(X_maint, y_maint)
            
            # Evaluate maintenance type prediction
            if len(maintenance_df) > 20:
                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                    X_maint, y_maint, test_size=0.2, random_state=42)
                y_pred_m = self.maintenance_type_model.predict(X_test_m)
                print("\nMaintenance Type Prediction Evaluation:")
                print(classification_report(y_test_m, y_pred_m))
    
    def predict_from_ids(self, vehicle_id, make, model, service_id, vehicles_df, services_df, fuel_logs_df, issues_df):
        """Predict maintenance needs based on vehicle and service IDs"""
        # Validate VIN format
        if len(vehicle_id) != 17 or not all(c in '0123456789ABCDEFGHJKLMNPRSTUVWXYZ' for c in vehicle_id.upper()):
            raise ValueError("Vehicle ID must be a 17-character alphanumeric VIN (excluding I, O, Q)")
            
        # Find the vehicle in the dataset
        try:
            vehicle = vehicles_df[vehicles_df['vehicle_id'] == vehicle_id].iloc[0]
        except IndexError:
            raise ValueError(f"Vehicle with VIN {vehicle_id} not found in database")
        
        # Get the specific service record
        try:
            service = services_df[services_df['service_id'] == service_id].iloc[0]
        except IndexError:
            raise ValueError(f"Service with ID {service_id} not found in database")
        
        # Verify the service belongs to this vehicle
        if service['vehicle_id'] != vehicle_id:
            raise ValueError(f"Service {service_id} does not belong to vehicle {vehicle_id}")
        
        # Calculate days since last service
        days_since_last_service = (datetime.now() - pd.to_datetime(service['service_date'])).days
        
        # Calculate mileage since last service
        mileage_since_last_service = vehicle['current_mileage'] - service['service_mileage']
        
        # Calculate service count in last 6 months
        service_count_6mo = len(services_df[
            (services_df['vehicle_id'] == vehicle_id) & 
            (pd.to_datetime(services_df['service_date']) > (datetime.now() - timedelta(days=180)))
        ])
        
        # Calculate average MPG for non-electric vehicles
        if vehicle['fuel_type'] != 'Electric' and not fuel_logs_df.empty:
            vehicle_fuel_logs = fuel_logs_df[fuel_logs_df['vehicle_id'] == vehicle_id]
            avg_mpg = vehicle_fuel_logs['mpg'].mean() if not vehicle_fuel_logs.empty else 25.0
        else:
            avg_mpg = 0.0
        
        # Prepare input features
        input_features = pd.DataFrame([{
            'make': make,
            'model': model,
            'year': vehicle['year'],
            'current_mileage': vehicle['current_mileage'],
            'fuel_type': vehicle['fuel_type'],
            'engine_type': vehicle['engine_type'],
            'days_since_last_service': days_since_last_service,
            'mileage_since_last_service': mileage_since_last_service,
            'avg_mpg': avg_mpg,
            'service_count_6mo': service_count_6mo
        }])
        
        # Encode categorical features
        for col in ['make', 'model', 'fuel_type', 'engine_type']:
            if col in input_features and col in self.label_encoders:
                input_features[col] = self.label_encoders[col].transform(input_features[col])
        
        # Make predictions
        needs_maintenance = self.model.predict(input_features)[0]
        maintenance_type = 'none'
        confidence = 0.0
        
        if needs_maintenance and hasattr(self, 'maintenance_type_model'):
            maintenance_type = self.maintenance_type_model.predict(input_features)[0]
            proba = self.maintenance_type_model.predict_proba(input_features)[0]
            confidence = max(proba)
        
        # Generate additional vehicle health metrics
        needs_oil_change = maintenance_type == 'oil_change'
        tire_wear_status = random.choice(['Good', 'Fair', 'Replace Soon'])
        battery_health = round(random.uniform(70, 100), 1)
        next_failure = random.choice(['None', 'Battery', 'Brakes', 'Tires', 'Engine'])
        failure_probability = round(random.uniform(0, 0.3), 2)
        
        return {
            'vehicle_id': vehicle_id,
            'needs_maintenance': bool(needs_maintenance),
            'maintenance_type': maintenance_type,
            'needs_oil_change': needs_oil_change,
            'tire_wear_status': tire_wear_status,
            'battery_health': battery_health,
            'next_failure': next_failure,
            'failure_probability': failure_probability,
            'confidence': float(confidence)
        }
    
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        joblib.dump({
            'model': self.model,
            'maintenance_type_model': getattr(self, 'maintenance_type_model', None),
            'label_encoders': self.label_encoders,
            'features': self.features,
            'target': self.target,
            'maintenance_types': self.maintenance_types
        }, filepath)
    
    def load_model(self, filepath):
        """Load a previously trained model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.maintenance_type_model = saved_data['maintenance_type_model']
        self.label_encoders = saved_data['label_encoders']
        self.features = saved_data['features']
        self.target = saved_data['target']
        self.maintenance_types = saved_data['maintenance_types']

def main():
    # Initialize predictor
    predictor = CarMaintenancePredictor()
    
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
    
    # Get sample vehicle and service IDs for demonstration
    sample_vehicle = vehicles_df.iloc[0]
    sample_service = services_df[services_df['vehicle_id'] == sample_vehicle['vehicle_id']].iloc[0]
    
    # Interactive prediction
    while True:
        print("\nEnter vehicle details for maintenance prediction:")
        print(f"Sample VIN: {sample_vehicle['vehicle_id']}")
        print(f"Sample Service ID: {sample_service['service_id']}")
        
        vehicle_id = input("Vehicle ID (17-character VIN): ").strip().upper() or sample_vehicle['vehicle_id']
        make = input(f"Make (e.g., {sample_vehicle['make']}): ").strip() or sample_vehicle['make']
        model = input(f"Model (e.g., {sample_vehicle['model']}): ").strip() or sample_vehicle['model']
        service_id = input(f"Service ID (e.g., {sample_service['service_id']}): ").strip() or sample_service['service_id']
        
        # Make prediction
        try:
            result = predictor.predict_from_ids(vehicle_id, make, model, service_id, 
                                             vehicles_df, services_df, fuel_logs_df, issues_df)
            
            # Display results
            print("\nMaintenance Prediction Results:")
            print(f"Vehicle ID: {result['vehicle_id']}")
            print(f"Maintenance Needed: {'Yes' if result['needs_maintenance'] else 'No'}")
            if result['needs_maintenance']:
                print(f"Maintenance Type: {result['maintenance_type'].replace('_', ' ').title()}")
                print(f"Confidence: {result['confidence']*100:.1f}%")
            
            print("\nVehicle Health Report:")
            print(f"Needs Oil Change: {'Yes' if result['needs_oil_change'] else 'No'}")
            print(f"Tire Wear Status: {result['tire_wear_status']}")
            print(f"Battery Health: {result['battery_health']}%")
            print(f"Next Likely Failure: {result['next_failure']}")
            print(f"Failure Probability: {result['failure_probability']*100:.0f}%")
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            print("Please check your input values and try again.")
        
        # Continue?
        cont = input("\nPredict another vehicle? (y/n): ").lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()