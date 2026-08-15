import os
import uuid
import secrets
from datetime import datetime
import pandas as pd
import streamlit as st

# Import your predictor model class
from model import CarMaintenancePredictor

# Set page layout configuration
st.set_page_config(
    page_title="Car Maintenance Predictor",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 1. State Initialization & Predictor Setup
# ------------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    """Load or train the model, cached across user sessions."""
    predictor = CarMaintenancePredictor()
    if os.path.exists('car_maintenance_model.pkl'):
        predictor.load_model('car_maintenance_model.pkl')
    else:
        vehicles_df, services_df, fuel_logs_df, issues_df = predictor.generate_synthetic_data()
        features_df = predictor.preprocess_data(vehicles_df, services_df, fuel_logs_df, issues_df)
        predictor.train_model(features_df)
        predictor.save_model("car_maintenance_model.pkl")
    return predictor

predictor = get_predictor()

# Initialize dynamic in-memory data tables within Streamlit Session State
if "data_loaded" not in st.session_state:
    v_df, s_df, f_df, i_df = predictor.generate_synthetic_data()
    st.session_state.vehicles = v_df.to_dict('records')
    st.session_state.services = s_df.to_dict('records')
    st.session_state.fuel_logs = f_df.to_dict('records') if not f_df.empty else []
    st.session_state.issues = i_df.to_dict('records') if not i_df.empty else []
    st.session_state.search_history = []
    st.session_state.data_loaded = True

# Helper Functions
def predict_maintenance_needed(vehicle_id):
    vehicle = next((v for v in st.session_state.vehicles if v['vehicle_id'] == vehicle_id), None)
    if not vehicle:
        return False
    
    vehicle_services = [s for s in st.session_state.services if s['vehicle_id'] == vehicle_id]
    if not vehicle_services:
        return True
    
    vehicle_services.sort(key=lambda x: str(x['service_date']), reverse=True)
    last_service = vehicle_services[0]
    
    try:
        s_date = datetime.strptime(str(last_service['service_date']), '%Y-%m-%d')
    except ValueError:
        s_date = datetime.now()
        
    days_since = (datetime.now() - s_date).days
    mileage_since = vehicle['current_mileage'] - last_service['service_mileage']
    
    return days_since > 180 or mileage_since > 7500

def get_prediction(vehicle_id):
    vehicle = next((v for v in st.session_state.vehicles if v['vehicle_id'] == vehicle_id), None)
    if not vehicle:
        return None
    
    vehicle_services = [s for s in st.session_state.services if s['vehicle_id'] == vehicle_id]
    if not vehicle_services:
        return None
    
    vehicle_services.sort(key=lambda x: str(x['service_date']), reverse=True)
    last_service = vehicle_services[0]
    
    try:
        v_df = pd.DataFrame(st.session_state.vehicles)
        s_df = pd.DataFrame(st.session_state.services)
        f_df = pd.DataFrame(st.session_state.fuel_logs) if st.session_state.fuel_logs else pd.DataFrame()
        i_df = pd.DataFrame(st.session_state.issues) if st.session_state.issues else pd.DataFrame()
        
        return predictor.predict_from_ids(
            vehicle_id, vehicle['make'], vehicle['model'], last_service['service_id'],
            v_df, s_df, f_df, i_df
        )
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# ------------------------------------------------------------------------------
# 2. Navigation Sidebar
# ------------------------------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Vehicles", "Services", "Predict Maintenance", "Add Vehicle", "Add Service"]
)

# ------------------------------------------------------------------------------
# 3. View: Dashboard
# ------------------------------------------------------------------------------
if menu == "Dashboard":
    st.title("🚗 Car Maintenance Predictor Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    vehicle_count = len(st.session_state.vehicles)
    service_count = len(st.session_state.services)
    maint_count = sum(1 for v in st.session_state.vehicles if predict_maintenance_needed(v['vehicle_id']))
    
    col1.metric("Total Vehicles", vehicle_count)
    col2.metric("Total Services", service_count)
    col3.metric("Maintenance Needed", maint_count)
    
    st.subheader("Vehicles Needing Maintenance")
    needing_maint = []
    for v in st.session_state.vehicles:
        if predict_maintenance_needed(v['vehicle_id']):
            v_info = v.copy()
            v_info['Prediction'] = get_prediction(v['vehicle_id'])
            needing_maint.append(v_info)
    
    if needing_maint:
        st.table(pd.DataFrame(needing_maint[:3]))
    else:
        st.info("No vehicles currently require immediate maintenance.")

    st.subheader("Recent Services")
    recent = sorted(st.session_state.services, key=lambda x: str(x['service_date']), reverse=True)[:5]
    st.dataframe(pd.DataFrame(recent), use_container_width=True)

# ------------------------------------------------------------------------------
# 4. View: Vehicles
# ------------------------------------------------------------------------------
elif menu == "Vehicles":
    st.title("📋 Vehicle List & Details")
    v_df = pd.DataFrame(st.session_state.vehicles)
    st.dataframe(v_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Inspect Specific Vehicle")
    selected_id = st.selectbox("Select VIN", v_df['vehicle_id'].unique() if not v_df.empty else [])
    
    if selected_id:
        vehicle = next((v for v in st.session_state.vehicles if v['vehicle_id'] == selected_id), None)
        if vehicle:
            st.write(f"**Make/Model:** {vehicle['make']} {vehicle['model']} ({vehicle['year']})")
            st.write(f"**Current Mileage:** {vehicle['current_mileage']} miles")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Service History")
                v_services = [s for s in st.session_state.services if s['vehicle_id'] == selected_id]
                st.dataframe(pd.DataFrame(v_services), use_container_width=True)
            with c2:
                st.subheader("Fuel Logs")
                v_fuels = [f for f in st.session_state.fuel_logs if f['vehicle_id'] == selected_id]
                st.dataframe(pd.DataFrame(v_fuels), use_container_width=True)

# ------------------------------------------------------------------------------
# 5. View: Services
# ------------------------------------------------------------------------------
elif menu == "Services":
    st.title("🛠 Service Records")
    st.dataframe(pd.DataFrame(st.session_state.services), use_container_width=True)

# ------------------------------------------------------------------------------
# 6. View: Predict Maintenance
# ------------------------------------------------------------------------------
elif menu == "Predict Maintenance":
    st.title("🔮 Maintenance Prediction")
    
    v_ids = [v['vehicle_id'] for v in st.session_state.vehicles]
    selected_v_id = st.selectbox("Vehicle VIN", v_ids)
    
    if selected_v_id:
        vehicle = next(v for v in st.session_state.vehicles if v['vehicle_id'] == selected_v_id)
        rel_services = [s['service_id'] for s in st.session_state.services if s['vehicle_id'] == selected_v_id]
        
        if rel_services:
            selected_s_id = st.selectbox("Service ID", rel_services)
            
            if st.button("Predict Maintenance Need"):
                result = get_prediction(selected_v_id)
                if result is not None:
                    st.success("Prediction generated successfully!")
                    st.write("### Result Output:")
                    st.json(result if isinstance(result, (dict, list)) else {"prediction": str(result)})
                else:
                    st.warning("Could not calculate prediction for selected IDs.")
        else:
            st.warning("No service records found for the selected vehicle.")

# ------------------------------------------------------------------------------
# 7. View: Add Vehicle
# ------------------------------------------------------------------------------
elif menu == "Add Vehicle":
    st.title("➕ Add New Vehicle")
    
    with st.form("add_vehicle_form"):
        make = st.text_input("Make", value="Toyota")
        model = st.text_input("Model", value="Camry")
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2020)
        mileage = st.number_input("Current Mileage", min_value=0, value=45000)
        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])
        engine_type = st.selectbox("Engine Type", ["V6", "I4", "V8", "Electric"])
        
        submitted = st.form_submit_button("Add Vehicle")
        if submitted:
            v_id = predictor.generate_vin() if hasattr(predictor, 'generate_vin') else f"VIN{secrets.token_hex(4).upper()}"
            new_vehicle = {
                'vehicle_id': v_id,
                'make': make,
                'model': model,
                'year': int(year),
                'current_mileage': int(mileage),
                'fuel_type': fuel_type,
                'engine_type': engine_type
            }
            st.session_state.vehicles.append(new_vehicle)
            st.success(f"Successfully added vehicle {make} {model} (ID: {v_id})!")

# ------------------------------------------------------------------------------
# 8. View: Add Service
# ------------------------------------------------------------------------------
elif menu == "Add Service":
    st.title("➕ Add Service Record")
    
    v_ids = [v['vehicle_id'] for v in st.session_state.vehicles]
    
    if not v_ids:
        st.warning("Please add a vehicle first.")
    else:
        with st.form("add_service_form"):
            v_id = st.selectbox("Select Vehicle", v_ids)
            service_type = st.text_input("Service Type", value="Oil Change")
            service_date = st.date_input("Service Date")
            service_mileage = st.number_input("Mileage at Service", min_value=0, value=50000)
            service_cost = st.number_input("Service Cost ($)", min_value=0.0, value=120.00)
            
            submitted = st.form_submit_button("Add Service Record")
            if submitted:
                s_id = f"SRV{len(st.session_state.services) + 10001}"
                new_service = {
                    'service_id': s_id,
                    'vehicle_id': v_id,
                    'service_type': service_type,
                    'service_date': str(service_date),
                    'service_mileage': int(service_mileage),
                    'service_cost': float(service_cost),
                    'next_service_mileage': int(service_mileage) + 5000
                }
                st.session_state.services.append(new_service)
                
                # Update vehicle mileage if higher
                vehicle = next(v for v in st.session_state.vehicles if v['vehicle_id'] == v_id)
                if int(service_mileage) > vehicle['current_mileage']:
                    vehicle['current_mileage'] = int(service_mileage)
                    
                st.success(f"Added service record {s_id} for Vehicle {v_id}!")
