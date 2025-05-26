# setup.py
import streamlit as st
from model.preprocess import load_and_preprocess
from model.train import train_model

@st.cache_resource
def setup_model():
    df_train, df_test = load_and_preprocess()
    model, vectorizer = train_model(df_train)
    return model, df_test, vectorizer
