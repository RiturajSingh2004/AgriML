import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

class CropAndFertilizerRecommender:
    def __init__(self):
        self.load_and_prepare_data()
        self.train_models()

    def load_and_prepare_data(self):
        # Load datasets
        fertilizer_df = pd.read_csv('Fertilizer Prediction.csv')
        crop_df = pd.read_csv('Crop_recommendation.csv')

        # Prepare fertilizer data
        self.fertilizer_data = pd.get_dummies(fertilizer_df,
                                            columns=['Soil Type', 'Crop Type'])
        self.fertilizer_X = self.fertilizer_data.drop('Fertilizer Name', axis=1)
        self.fertilizer_y = self.fertilizer_data['Fertilizer Name']

        # Prepare crop data
        self.crop_X = crop_df.drop('label', axis=1)
        self.crop_y = crop_df['label']

        # Store original dataframes for later use
        self.fertilizer_df = fertilizer_df
        self.crop_df = crop_df

    def train_models(self):
        # Fertilizer Model
        X_train_fert, X_test_fert, y_train_fert, y_test_fert = train_test_split(
            self.fertilizer_X, self.fertilizer_y, test_size=0.2, random_state=42
        )

        self.scaler_fert = StandardScaler()
        X_train_fert_scaled = self.scaler_fert.fit_transform(X_train_fert)

        self.fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.fertilizer_model.fit(X_train_fert_scaled, y_train_fert)

        # Crop Model
        X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
            self.crop_X, self.crop_y, test_size=0.2, random_state=42
        )

        self.scaler_crop = StandardScaler()
        X_train_crop_scaled = self.scaler_crop.fit_transform(X_train_crop)

        self.crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.crop_model.fit(X_train_crop_scaled, y_train_crop)

    def recommend_fertilizer(self, temperature, humidity, moisture,
                           soil_type, crop_type, nitrogen, potassium, phosphorous):
        input_data = pd.DataFrame({
            'Temparature': [temperature],
            'Humidity': [humidity],
            'Moisture': [moisture],
            'Soil Type': [soil_type],
            'Crop Type': [crop_type],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorous]
        })

        input_encoded = pd.get_dummies(input_data)

        for col in self.fertilizer_X.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded.reindex(columns=self.fertilizer_X.columns, fill_value=0)
        input_scaled = self.scaler_fert.transform(input_encoded)

        prediction = self.fertilizer_model.predict(input_scaled)
        proba = self.fertilizer_model.predict_proba(input_scaled)
        return prediction[0], proba[0]

    def recommend_crop(self, nitrogen, phosphorous, potassium,
                      temperature, humidity, ph, rainfall):
        input_data = np.array([[nitrogen, phosphorous, potassium,
                               temperature, humidity, ph, rainfall]])

        input_scaled = self.scaler_crop.transform(input_data)

        prediction = self.crop_model.predict(input_scaled)
        proba = self.crop_model.predict_proba(input_scaled)
        return prediction[0], proba[0]

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
        }
    ))
    fig.update_layout(height=200)
    return fig

def main():
    st.set_page_config(page_title="Agricultural Recommendation System", layout="wide")

    st.title("🌱 Agricultural Recommendation System")

    # Initialize the recommender
    @st.cache_resource
    def load_recommender():
        return CropAndFertilizerRecommender()

    try:
        recommender = load_recommender()

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Fertilizer Recommendation", "Crop Recommendation", "Data Insights"])

        # Fertilizer Recommendation Tab
        with tab1:
            st.header("Fertilizer Recommendation")

            col1, col2 = st.columns(2)

            with col1:
                temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
                humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
                moisture = st.slider("Moisture (%)", 0.0, 100.0, 50.0)
                soil_type = st.selectbox("Soil Type", recommender.fertilizer_df['Soil Type'].unique())

            with col2:
                crop_type = st.selectbox("Crop Type", recommender.fertilizer_df['Crop Type'].unique())
                nitrogen = st.slider("Nitrogen Level", 0, 140, 50)
                phosphorous = st.slider("Phosphorous Level", 0, 140, 50)
                potassium = st.slider("Potassium Level", 0, 140, 50)

            if st.button("Get Fertilizer Recommendation"):
                prediction, probabilities = recommender.recommend_fertilizer(
                    temperature, humidity, moisture, soil_type, crop_type,
                    nitrogen, potassium, phosphorous
                )

                st.success(f"### Recommended Fertilizer: {prediction}")

                # Display confidence
                confidence = max(probabilities) * 100
                st.plotly_chart(create_gauge_chart(confidence, "Recommendation Confidence (%)"))

                # Display input summary
                st.write("### Input Parameters Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Temperature: {temperature}°C")
                    st.write(f"Humidity: {humidity}%")
                    st.write(f"Moisture: {moisture}%")
                    st.write(f"Soil Type: {soil_type}")
                with col2:
                    st.write(f"Crop Type: {crop_type}")
                    st.write(f"Nitrogen: {nitrogen}")
                    st.write(f"Phosphorous: {phosphorous}")
                    st.write(f"Potassium: {potassium}")

        # Crop Recommendation Tab
        with tab2:
            st.header("Crop Recommendation")

            col1, col2 = st.columns(2)

            with col1:
                nitrogen_crop = st.slider("Nitrogen Level ", 0, 140, 50)
                phosphorous_crop = st.slider("Phosphorous Level ", 0, 140, 50)
                potassium_crop = st.slider("Potassium Level ", 0, 140, 50)
                temperature_crop = st.slider("Temperature (°C) ", 0.0, 50.0, 25.0)

            with col2:
                humidity_crop = st.slider("Humidity (%) ", 0.0, 100.0, 50.0)
                ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
                rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

            if st.button("Get Crop Recommendation"):
                prediction, probabilities = recommender.recommend_crop(
                    nitrogen_crop, phosphorous_crop, potassium_crop,
                    temperature_crop, humidity_crop, ph, rainfall
                )

                st.success(f"### Recommended Crop: {prediction}")

                # Display confidence
                confidence = max(probabilities) * 100
                st.plotly_chart(create_gauge_chart(confidence, "Recommendation Confidence (%)"))

                # Display input summary
                st.write("### Input Parameters Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Nitrogen: {nitrogen_crop}")
                    st.write(f"Phosphorous: {phosphorous_crop}")
                    st.write(f"Potassium: {potassium_crop}")
                    st.write(f"Temperature: {temperature_crop}°C")
                with col2:
                    st.write(f"Humidity: {humidity_crop}%")
                    st.write(f"pH: {ph}")
                    st.write(f"Rainfall: {rainfall} mm")

        # Data Insights Tab
        with tab3:
            st.header("Data Insights")

            # Show distribution of crops and fertilizers
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Crop Distribution")
                fig = px.pie(recommender.crop_df, names='label', title='Distribution of Crops in Dataset')
                st.plotly_chart(fig)

            with col2:
                st.subheader("Fertilizer Distribution")
                fig = px.pie(recommender.fertilizer_df, names='Fertilizer Name',
                           title='Distribution of Fertilizers in Dataset')
                st.plotly_chart(fig)

            # Show correlation heatmap for crop data
            st.subheader("Crop Features Correlation")
            crop_corr = recommender.crop_X.corr()
            fig = px.imshow(crop_corr, title='Correlation between Crop Features',
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please ensure all required files are present and try again.")

if __name__ == "__main__":
    main()
