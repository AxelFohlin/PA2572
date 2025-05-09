import streamlit as st
from setup import setup_model

from model.evaluation import evaluate_model, display_feature_importance
#asd
model, df_test, vectorizer = setup_model()
evaluation_results, fig = evaluate_model(model, df_test, vectorizer)
feature_importance_df = display_feature_importance(model, vectorizer)

st.header("Model Statistics")

st.subheader("Evaluation statistics")
st.write(evaluation_results)
st.pyplot(fig)

st.subheader("Feature Importance")
st.dataframe(feature_importance_df)