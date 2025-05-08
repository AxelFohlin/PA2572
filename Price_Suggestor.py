import streamlit as st
from model.preprocess import load_and_preprocess
from model.train import train_model
from model.predict import prepare_features, predict_price
from streamlit_tags import st_tags
from streamlit_geolocation import streamlit_geolocation

if 'property_info' not in st.session_state:
    st.session_state['property_info'] = {
        'amenities': [],
        'bedrooms': 0,
        'bathrooms': 0.0,
        'accommodates': 0,
        'minimum_nights': 0
    }

if "amenity_list" not in st.session_state:
    st.session_state["amenity_list"] = []

# Cache the model loading and training
@st.cache_resource
def setup():
    # yo
    df, vectorizer = load_and_preprocess()
    model = train_model(df, vectorizer)
    return model, vectorizer

model, vectorizer = setup()

with st.sidebar:
    pass


st.title("Airbnb Price Predictor")

amenities = st_tags(
    label='Enter Amenities:',
    text='Press enter to add more',
    value=[],
    maxtags = -1,
    key='1')

# amenities = ["Washer", "Hair dryer", "Hangers", "Indoor fireplace", "Wifi", "Exterior security cameras on property", "Iron", "Kitchen", "Free parking on premises", "Smoke alarm", "Fire extinguisher", "Dryer", "TV", "Heating", "Hot water", "Shampoo", "Cooking basics", "Patio or balcony", "Dedicated workspace", "Refrigerator", "Backyard", "Coffee maker"]

location = streamlit_geolocation()
if location:
    st.badge(f'LAT: {location["latitude"]}', color="green")
    st.badge(f'LON: {location["longitude"]}', color="green")
else:
    st.write("Please provide your location.")

bedrooms = st.number_input("Bedrooms", min_value=0, value=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0, step=0.5)
accommodates = st.number_input("Accommodates", min_value=1, value=2)
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=2)
maximum_nights = st.number_input("Minimum Nights", min_value=1, value=4)

if st.button("Predict Price"):
    numerical = [bedrooms, bathrooms, accommodates, minimum_nights, maximum_nights, location['longitude'], location['latitude']]

    features = prepare_features(vectorizer, amenities, numerical)
    price = predict_price(model, features)

    st.session_state['property_info'] = {
        'amenities': amenities,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'accommodates': accommodates,
        'minimum_nights': minimum_nights
    }

    st.success(f"💰 Estimated Price per Night: **${price:.2f}**")
