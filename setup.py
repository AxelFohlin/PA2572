# setup.py
import streamlit as st
from model.preprocess import load_and_preprocess
from model.train import train_model

@st.cache_resource
def setup_model():
    df_train, df_test, vectorizer = load_and_preprocess()
    model = train_model(df_train, vectorizer)
    return model, df_test, vectorizer
