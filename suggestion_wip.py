# review_insights.py

import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import ast

from sklearn.feature_extraction.text import CountVectorizer
# Ensure stopwords are downloaded

custom_stopwords = [
    "great", "place", "stay", "stockholm", "nice", "location",
    "host", "room", "apartment", "recommend", "really", "good",
    "would", "everything", "also", "perfect", "well", "tommy", "old", "city", "town", "check",
    "easy", "flat", "stan", "gamla", "located", "highly", "definitely", "time", "br"
]

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.update(custom_stopwords)

stop_words = list(stop_words)

def get_keywords_by_location_and_type(
    df_reviews: pd.DataFrame,
    df_listings: pd.DataFrame,
    neighbourhood_cleansed: str,
    room_type: str,
    top_n: int = 20,
    text_source : str = "comments"
) -> list[tuple[str, int]]:
    """
    Extracts the top positive review keywords for a specific neighborhood and room type.

    Parameters:
        df_reviews: DataFrame with sentiment-labeled reviews (must include "listing_id", "sentiment", "comments")
        df_listings: DataFrame with listing metadata (must include "id", "neighbourhood_cleansed", "room_type")
        neighborhood: Selected neighborhood (e.g. "Södermalm")
        room_type: Selected room type (e.g. "Private room")
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, count) tuples
    """

    # Merge reviews with listing metadata
    merged = df_reviews.merge(
        df_listings[["id", "neighbourhood_cleansed", "room_type", "name", "amenities"]],
        left_on="listing_id",
        right_on="id",
        how="inner"
    )
    print(merged.columns)
    # Filter to positive reviews in selected context
    filtered = merged[
        (merged["sentiment"] == "POSITIVE") &
        (merged["neighbourhood_cleansed"] == neighbourhood_cleansed) &
        (merged["room_type"] == room_type)
    ]

    if filtered.empty:
        return []

    # Tokenize and count words
    text_source = filtered[text_source].dropna().str.lower().tolist()
    all_text = " ".join(text_source)
    tokens = re.findall(r"\b[a-z]{3,}\b", all_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return Counter(filtered_tokens).most_common(top_n)


def get_keywords_by_location_and_type_test(
    df_reviews: pd.DataFrame,
    df_listings: pd.DataFrame,
    neighbourhood_cleansed: str,
    room_type: str,
    top_n: int = 20,
    text_source: str = "comments",
    ngram_range: tuple = (1, 2),
) -> list[tuple[str, int]]:

    # Merge relevant listing metadata
    merged = df_reviews.merge(
        df_listings[["id", "neighbourhood_cleansed", "room_type", "name", "amenities", "description"]],
        left_on="listing_id",
        right_on="id",
        how="inner"
    )

    # Filter by neighborhood and room type
    filtered = merged[
        (merged["neighbourhood_cleansed"] == neighbourhood_cleansed) &
        (merged["room_type"] == room_type)
    ]

    # Only use POSITIVE reviews when analyzing comments
    filtered = filtered[filtered["sentiment"] == "POSITIVE"]

    if filtered.empty:
        return []

    # Special handling for amenities
    if text_source == "amenities":
        all_amenities = []
        for entry in filtered["amenities"].dropna():
            try:
                items = ast.literal_eval(entry)
                all_amenities.extend([item.lower() for item in items])
            except:
                continue
        tokens = [token for token in all_amenities if token not in stop_words]
        return Counter(tokens).most_common(top_n)

    # Handle title, description, or comments using CountVectorizer

    # 1. Prepare the selected text column (e.g. 'comments', 'name', or 'description') 
    # - Remove missing values
    # - Convert to string
    # - Convert to a list of documents (each row becomes one text)
    text_data = filtered[text_source].dropna().astype(str).tolist()

    # 2. Initialize a CountVectorizer for extracting word/phrase counts
    # - stop_words: remove common and domain-specific words
    # - ngram_range: choose whether to include 1-word, 2-word, or 3-word phrases
    # - lowercase: make everything lowercase
    # - max_features: optional cap on number of features to extract
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        lowercase=True,
        max_features=1000
    )

    # 3. Transform the text data into a sparse matrix of n-gram counts
    # Each column in X corresponds to an n-gram (word or phrase)
    # Each row corresponds to one text entry (e.g., one review)
    X = vectorizer.fit_transform(text_data)

    # 4. Sum the counts for each n-gram across all documents
    # This gives the total number of times each phrase appeared
    counts = X.sum(axis=0).A1

    # 5. Get the actual n-gram strings (e.g., "great location", "clean apartment")
    vocab = vectorizer.get_feature_names_out()

    # 6. Combine each phrase with its corresponding count
    # Then sort from most frequent to least frequent
    phrase_counts = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)

    return phrase_counts[:top_n]

df_reviews = pd.read_csv("data/sentiment_reviews.csv")
df_listings = pd.read_csv("data/listings/listings.csv")

print(df_listings["name"].unique())

# ['Södermalms' 'Kungsholmens' 'Norrmalms' 'Hägersten-Liljeholmens' 'Älvsjö'
#  'Skarpnäcks' 'Farsta' 'Enskede-Årsta-Vantörs' 'Östermalms' 'Bromma'
#  'Skärholmens' 'Hässelby-Vällingby' 'Spånga-Tensta' 'Rinkeby-Tensta']

# ['Private room' 'Entire home/apt' 'Shared room' 'Hotel room']

keywords = get_keywords_by_location_and_type_test(
    df_reviews,
    df_listings,
    neighbourhood_cleansed="Södermalms",
    room_type="Entire home/apt",
    top_n=10,
    text_source="description",
    ngram_range=(2,2)
)
# name -> title
# comments -> reviews
# desciption -> Host

# TODO: Clean description breaks!!

# (1, 1) -> unigrams "clean", "nice"
# (1, 2) -> unigrams and bigrams "clean", "great location"

print("📢 Popular keywords for selected options:")
for word, count in keywords:
    print(f"✅ {word} ({count})")


room_counts = df_listings.groupby(['neighbourhood_cleansed', 'room_type']).size().reset_index(name='count')

room_counts_pivot = room_counts.pivot(index='neighbourhood_cleansed', columns='room_type', values='count').fillna(0).astype(int)

# Display
print(room_counts_pivot)
