import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Marriage Prediction AI",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and encoders
@st.cache_resource
def load_resources():
    model = load_model('marriage_model.h5')
    with open('serialized_object/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('serialized_object/onehot_encoder.pkl', 'rb') as file:
        onehot_encoder = pickle.load(file)
    with open('serialized_object/scalar.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder, onehot_encoder, scaler

model, label_encoder, onehot_encoder, scaler = load_resources()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>💍 Marriage Prediction AI</h1>
        <p>Discover the likelihood of a successful marriage with AI-powered predictions</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = option_menu(None, ["Prediction", "About"], 
                       icons=['heart-pulse', 'info-circle'],
                       menu_icon="cast", default_index=0)

if page == "Prediction":
    st.markdown("### 📝 Enter Couple Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Person 1")
        age1 = st.number_input("Age", min_value=18, max_value=80, value=28, key="age1")
        gender1 = st.selectbox("Gender", ["Male", "Female"], key="gender1")
        religion1 = st.selectbox("Religion", 
                                 ["Hindu", "Muslim", "Christian", "Buddhist", "Sikh", "Jain", "Parsi"],
                                 key="religion1")
        location1 = st.selectbox("Location Type", 
                                ["Rural", "Semi-Urban", "Urban"], key="location1")
        intercaste = st.radio("Intercaste Marriage?", ["No", "Yes"], horizontal=True)
    
    with col2:
        st.markdown("#### 💑 Person 2")
        age2 = st.number_input("Age", min_value=18, max_value=80, value=27, key="age2")
        gender2 = st.selectbox("Gender", ["Male", "Female"], key="gender2")
        religion2 = st.selectbox("Religion", 
                                 ["Hindu", "Muslim", "Christian", "Buddhist", "Sikh", "Jain", "Parsi"],
                                 key="religion2")
        location2 = st.selectbox("Location Type", 
                                ["Rural", "Semi-Urban", "Urban"], key="location2")
    
    # Additional details
    st.markdown("#### 📊 Relationship Details")
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        years_together = st.number_input("Years Together", min_value=0.0, max_value=50.0, value=3.5, step=0.5)
    
    with col4:
        compatibility = st.slider("Compatibility Score", 0, 100, 75)
    
    with col5:
        happy = st.radio("Are They Happy?", ["No", "Yes"], horizontal=True)
    
    with col6:
        want_marry = st.radio("Want to Marry?", ["No", "Yes"], horizontal=True)
    
    # Prediction button
    if st.button("🔮 Predict Marriage Success", use_container_width=True):
        try:
            # Prepare data
            age_diff = abs(age1 - age2)
            
            # Create sample data dictionary
            sample_data = {
                'Age': age1,
                'Gender': gender1,
                'Partner_Age': age2,
                'Partner_Gender': gender2,
                'Years_Together': years_together,
                'Age_Difference': age_diff,
                'Intercaste': intercaste,
                'Are_They_Happy': happy,
                'Want_To_Marry': want_marry,
                'Compatibility_Score': compatibility,
                'Religion': religion1,
                'Partner_Religion': religion2,
                'Location_Type': location1,
                'Partner_Location_Type': location2,
            }
            
            # Encode categorical features
            sample_data['Gender'] = label_encoder.fit_transform([sample_data['Gender']])[0]
            sample_data['Partner_Gender'] = label_encoder.fit_transform([sample_data['Partner_Gender']])[0]
            sample_data['Intercaste'] = label_encoder.fit_transform([sample_data['Intercaste']])[0]
            sample_data['Are_They_Happy'] = label_encoder.fit_transform([sample_data['Are_They_Happy']])[0]
            sample_data['Want_To_Marry'] = label_encoder.fit_transform([sample_data['Want_To_Marry']])[0]
            
            # One-hot encode
            categorical_data = pd.DataFrame([[
                religion1, religion2, location1, location2
            ]], columns=['Religion', 'Partner_Religion', 'Location_Type', 'Partner_Location_Type'])
            
            encoded_categorical = onehot_encoder.transform(categorical_data).toarray()
            
            # Create final input
            final_input = np.array([[
                sample_data['Age'],
                sample_data['Gender'],
                sample_data['Partner_Age'],
                sample_data['Partner_Gender'],
                sample_data['Years_Together'],
                sample_data['Age_Difference'],
                sample_data['Intercaste'],
                sample_data['Are_They_Happy'],
                sample_data['Want_To_Marry'],
                sample_data['Compatibility_Score'],
                *encoded_categorical[0]
            ]])
            
            # Scale and predict
            final_input_scaled = scaler.transform(final_input)
            prediction = model.predict(final_input_scaled, verbose=0)
            probability = prediction[0][0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if probability > 0.5:
                    st.markdown(f"""
                        <div class="prediction-positive">
                            <h3>✅ Highly Likely to Marry!</h3>
                            <h2>{probability*100:.1f}%</h2>
                            <p>The AI predicts this couple has a <b>strong chance</b> of marriage</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-negative">
                            <h3>⚠️ Lower Probability</h3>
                            <h2>{probability*100:.1f}%</h2>
                            <p>The AI suggests this couple may face challenges before marriage</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_result2:
                # Display couple summary
                st.markdown("#### 👥 Couple Summary")
                st.info(f"""
                **YOU:** {age1}-year-old {gender1.lower()} ({religion1}, {location1})
                
                **YOUR PARTNER:** {age2}-year-old {gender2.lower()} ({religion2}, {location2})
                
                **Relationship:** {years_together} years together
                **Compatibility:** {compatibility}/100
                **Happy:** {'Yes' if happy == 'Yes' else 'No'}
                **Want to Marry:** {'Yes' if want_marry == 'Yes' else 'No'}
                """)
            
            # Display confidence gauge
            st.markdown("#### 📊 Confidence Metrics")
            col_conf1, col_conf2 = st.columns(2)
            
            with col_conf1:
                st.metric("Marriage Probability", f"{probability*100:.2f}%")
            
            with col_conf2:
                st.metric("Confidence Level", f"{max(probability, 1-probability)*100:.2f}%")
        
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")

elif page == "About":
    st.markdown("### ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🤖 How It Works
        
        This AI model uses an Artificial Neural Network (ANN) trained on marriage data 
        to predict the likelihood of a couple getting married based on various factors.
        
        ### 📈 Model Details
        - **Algorithm:** Deep Neural Network (Keras/TensorFlow)
        - **Training Data:** 10,000+ marriage records
        - **Accuracy:** High precision classification
        """)
    
    with col2:
        st.markdown("""
        ## 🎯 Factors Considered
        
        The model analyzes:
        - 👥 **Individual Traits:** Age, Gender
        - 💑 **Relationship Factors:** Years together, Compatibility
        - 🏠 **Social Factors:** Religion, Location type
        - ✨ **Emotional Factors:** Happiness level, Marriage desire
        """)
    
    st.markdown("---")
    st.markdown("""
    ### ✨ Disclaimer
    This is an AI-powered prediction tool for entertainment and analytical purposes. 
    Actual marriage decisions depend on many personal factors beyond data analysis.
    """, unsafe_allow_html=True)