import streamlit as st
import pandas as pd
from suggestions import get_keywords_by_location_and_type
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

generator = pipeline("text2text-generation", model="google/flan-t5-small")

df_reviews = pd.read_csv("data/sentiment_reviews.csv")
df_listings = pd.read_csv("data/listings/listings.csv")

if 'keywords' not in st.session_state:
    st.session_state['keywords'] = []

tab1, tab2 = st.tabs(["Keyword Suggestor", "Generate Title"])

with tab1:
    st.title("Keyword Suggestor")

    neighborhood = st.selectbox("Select Neighborhood", sorted(df_listings['neighbourhood_cleansed'].dropna().unique()))
    room_type = st.selectbox("Select Room Type", sorted(df_listings['room_type'].dropna().unique()))
    top_n = st.slider("Number of Top Keywords", min_value=5, max_value=20, value=5, step=5)

    if st.button("Suggest Keywords"):
        keywords = get_keywords_by_location_and_type(
            df_reviews=df_reviews,
            df_listings=df_listings,
            neighborhood=neighborhood,
            room_type=room_type,
            top_n=top_n
        )
        st.session_state['keywords'] = keywords

        st.subheader("Top Suggested Keywords")
        for i, (keyword, count) in enumerate(keywords, 1):
            st.write(f"**{i}.** :blue-background[{keyword}]:gray[ — **{count:,}** mentions]")


with tab2:
    if st.button("Generate Title", disabled="keywords" not in st.session_state or "property_info" not in st.session_state):
        keywords = [kw[0] for kw in st.session_state.keywords[:5]]
        prop = st.session_state.property_info

        # Access stored values
        amenities = prop["amenities"]
        bedrooms = prop["bedrooms"]
        bathrooms = prop["bathrooms"]
        accommodates = prop["accommodates"]

        # Use in prompt
        prompt = (
            f"Can you generate a catchy title for a airbnb property with {bedrooms} bedroom(s)")
        # prompt = (
        #     f"Can you generate a catchy Airbnb title for a property with {bedrooms} bedroom(s), "
        #     f"{bathrooms} bathroom(s), accommodates {accommodates} guests, "
        #     f"has amenities like {', '.join(amenities[:3])}, and is described with words like {', '.join(keywords[:5])}."
        # )

        st.write("### Prompt")
        st.write(f"Prompt: {prompt}")

        result = generator(prompt)

        st.write("### Generated Title")
        st.write(result)
        
