import streamlit as st
import pandas as pd

from setup import setup_model
from model.predict import prepare_features, predict_price
from streamlit_tags import st_tags
from streamlit_geolocation import streamlit_geolocation
from model.evaluation import display_feature_importance
from suggestions import get_keywords_count

df_reviews = pd.read_csv("data/sentiment_reviews.csv")
df_listings = pd.read_csv("data/listings/listings.csv")

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

model, df_test, vectorizer = setup_model()

with st.sidebar:
    pass


st.title("Airbnb Price Predictor")

amenities = st_tags(
    label='Enter Amenities:',
    text='Press enter to add more',
    value=[],
    maxtags = -1,
    key='1')


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
maximum_nights = st.number_input("Maximum Nights", min_value=1, value=4)

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

    st.divider()

    st.success(f"💰 Estimated Price per Night: **{price:.2f} SEK**")

    col1, col2, col3 = st.columns(3, border=True)

    with col1:
        st.write("Important Keywords")
        imp_keywords_title = get_keywords_count(
            df_reviews=df_reviews,
            df_listings=df_listings,
            neighbourhood_cleansed="all",
            room_type="all",
            top_n=10,
            text_source="name",
        )
        imp_keywords_description = get_keywords_count(
            df_reviews=df_reviews,
            df_listings=df_listings,
            neighbourhood_cleansed="all",
            room_type="all",
            top_n=10,
            text_source="description",
        )

        badge_string = " ".join([f":green-badge[{kw}]" for kw, _ in imp_keywords_title])
        st.markdown(":gray[Title keywords]")
        st.markdown(badge_string)

        badge_string = " ".join([f":green-badge[{kw}]" for kw, _ in imp_keywords_description])
        st.markdown(":gray[Description keywords]")
        st.markdown(badge_string)
    with col2:
        st.write("High-Impact Words")
        keywords = get_keywords_count(
            df_reviews=df_reviews,
            df_listings=df_listings,
            neighbourhood_cleansed="all",
            room_type="all",
            top_n=10,
            text_source="comments",
        )

        badge_string = " ".join([f":blue-badge[{kw}]" for kw, _ in keywords])
        st.markdown(badge_string)
    with col3:
        st.write("Competitive Amenities")
        feature_importance_df = display_feature_importance(model, vectorizer)
        excluded_features = {"amenities", "bedrooms", "bathrooms", "accommodates", "minimum_nights", "longitude", "latitude", "maximum_nights"}

        filtered_df = feature_importance_df[~feature_importance_df["Feature"].isin(excluded_features)]
        filtered_df = filtered_df.sort_values(by="Importance", ascending=False)

        badge_string = " ".join([f":orange-badge[{row['Feature']}]" for _, row in filtered_df.iterrows()])
        st.markdown(badge_string)