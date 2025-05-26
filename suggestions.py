import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from keybert import KeyBERT
import ast

from sklearn.feature_extraction.text import CountVectorizer

df_reviews = pd.read_csv("data/sentiment_reviews.csv")
df_listings = pd.read_csv("data/listings/listings.csv")

CUSTOM_STOPWORDS = [
    "med", "södermalm", "Södermalm", "södermalms", "Södermalms", "stockholm", "Stockholm", "stockholms", "Stockholms", "apartment", "apartments", "och", "på", "br", "the",
    "stockholmcity", "could", "sweden", "swedish", "staockholm", "sverige",
    "sjöstad", "hökmossen", "lund", "hostels", "hostel", "kungsholmstorg", "liljeholmen", "medborgerplatsen", "judarskogen", "baggensgatan"
, "långholmen", "medborgarplatsen", "skånegatan", "djurgården", "haha"]

NGRAM_RANGE = (1,1)

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update(CUSTOM_STOPWORDS)

stop_words = list(stop_words)

def get_keywords_count(
    df_reviews: pd.DataFrame,
    df_listings: pd.DataFrame,
    neighbourhood_cleansed: str = "all",
    room_type: str = "all",
    top_n: int = 10,
    text_source: str = "description",
) -> list[tuple[str, int]]:

    # Keep only listings that have positive reviews
    positive_reviews = df_reviews[df_reviews["sentiment"] == "POSITIVE"]
    positive_ids = positive_reviews["listing_id"].unique()
    df_listings = df_listings[df_listings["id"].isin(positive_ids)]

    # Filter by neighborhood and room type
    filtered = df_listings.copy()
    if neighbourhood_cleansed.lower() != "all":
        filtered = filtered[filtered["neighbourhood_cleansed"] == neighbourhood_cleansed]
    if room_type.lower() != "all":
        filtered = filtered[filtered["room_type"] == room_type]

    if filtered.empty:
        return []
    
    filtered = filtered.merge(
        positive_reviews[["listing_id", "comments"]],
        how="left",
        left_on="id",
        right_on="listing_id"
    )

    # Handle name, description, etc. using CountVectorizer
    text_data = filtered[text_source].dropna().astype(str).tolist()

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=NGRAM_RANGE,
        lowercase=True,
        max_features=1000
    )

    X = vectorizer.fit_transform(text_data)
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    phrase_counts = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)

    return phrase_counts[:top_n]


def get_keywords_BERT(
    df_listings: pd.DataFrame,
    df_reviews: pd.DataFrame,
    text_column: str = "description",
    top_n: int = 10,
    model_name: str = "all-mpnet-base-v2", # all-MiniLM-L6-v2 # paraphrase-MiniLM-L12-v2 # all-mpnet-base-v2 
    neighbourhood: str = "all",
    room_type: str = "all",
    stopwords: list[str] = None,
    ngram_range: tuple = (1, 2)
) -> list[tuple[str, float]]:

    kw_model = KeyBERT(model_name)
    if text_column == "amenities":
        df_listings["amenities_text"] = df_listings["amenities"].fillna("[]").apply(
        lambda x: " ".join(ast.literal_eval(x)).lower())

    positive_ids = df_reviews[df_reviews["sentiment"] == "POSITIVE"]["listing_id"].unique()
    df_listings = df_listings[df_listings["id"].isin(positive_ids)]

    # Optional filtering by neighbourhood / room type
    if neighbourhood.lower() != "all":
        df_listings = df_listings[df_listings["neighbourhood_cleansed"] == neighbourhood]
    if room_type.lower() != "all":
        df_listings = df_listings[df_listings["room_type"] == room_type]

    # Clean and combine text
    text_series = df_listings[text_column].dropna().astype(str)
    print("text_series: ", text_series.shape)
    # if len(text_series) > 1600:
    #     text_series = text_series.sample(n=1600, random_state=42)
    #     print("text_series sample: ", text_series.shape)
    if text_series.empty:
        return []
    combined_text = " ".join(text_series.tolist())

    # Extract keywords
    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=ngram_range,
        stop_words=stopwords,
        top_n=top_n,
        # use_maxsum=True,
        # nr_candidates=25
        # -----
        # use_mmr=True,
        # diversity=0.5
    )
    return keywords
