import streamlit as st
from sentence_transformers import SentenceTransformer
from model.preprocess import load_and_preprocess
from model.train import train_model
from model.predict import prepare_features, predict_price

# Cache the model loading and training
@st.cache_resource
def setup():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    df = load_and_preprocess(embedding_model)
    model = train_model(df)
    return embedding_model, model

with st.spinner("Loading model and preprocessing data..."):
    embedding_model, model = setup()

st.title("🧠 Airbnb Price Predictor")

amenities = st.multiselect(
    "Select amenities",
    ["Hair dryer", "Hangers", "Long term stays allowed", "Host greets you", "Bathtub",
     "Luggage dropoff allowed", "Iron", "Essentials", "Free washer – In building",
     "Elevator", "Free dryer – In building", "Courtyard view", "Smoke alarm", "TV",
     "Garden view", "Dishes and silverware", "Shared backyard – Not fully fenced",
     "Outdoor playground", "Heating", "Hot water", "Shampoo", "Bed linens",
     "Extra pillows and blankets", "Lock on bedroom door", "Fast wifi – 399 Mbps",
     "Park view", "Refrigerator", "Microwave", "Coffee maker"]
)

bedrooms = st.number_input("Bedrooms", min_value=0, value=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0)
accommodates = st.number_input("Accommodates", min_value=1, value=2)
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=2)

if st.button("Predict Price"):
    numerical = [bedrooms, bathrooms, accommodates, minimum_nights]
    features = prepare_features(embedding_model, amenities, numerical)
    price = predict_price(model, features)
    st.success(f"💰 Estimated Price per Night: **${price:.2f}**")
