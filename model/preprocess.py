import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

def append_sentiment_score(listings_df: pd.DataFrame, reviews_df: pd.DataFrame):
    """
    Add sentiment score directly to the listings_df based on reviews_df.
    This modifies listings_df in place.
    """
    listings_df['id'] = listings_df['id'].astype(int)
    reviews_df['listing_id'] = reviews_df['listing_id'].astype(int)

    # Convert sentiment labels to binary scores
    reviews_df['sentiment_score'] = reviews_df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})

    # Aggregate average sentiment score
    aggregated_sentiment = (
        reviews_df.groupby('listing_id')['sentiment_score']
        .mean()
        .reset_index()
    )
    sentiment_map = dict(zip(aggregated_sentiment['listing_id'], aggregated_sentiment['sentiment_score']))
    listings_df['sentiment_score'] = listings_df['id'].map(sentiment_map)

def tfidf_amenities(listings_df: pd.DataFrame, max_features=100):
    # Convert stringified lists to space-separated strings
    listings_df["amenities_clean"] = listings_df["amenities"].fillna("[]").apply(
        lambda x: " ".join(ast.literal_eval(x))
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(listings_df["amenities_clean"])
    
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    listings_df.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    
    listings_df = pd.concat([listings_df, tfidf_df], axis=1)
    return listings_df, vectorizer

def load_and_preprocess():
    data = pd.read_csv("data/listings/listings.csv")
    important_columns = [
        'id', 'price', 'neighbourhood_cleansed', 'room_type',
        'bedrooms', 'bathrooms', 'accommodates', 'amenities',
        'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating',
        'longitude', 'latitude', 'name', 'description'
    ]
    df = data[important_columns].copy()
    df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)
    df = df.dropna(subset=['price'])
    df = df[df['price'] != 0]

    lower = df['price'].quantile(0.05)
    upper = df['price'].quantile(0.95)
    df = df[(df['price'] >= lower) & (df['price'] <= upper)]

    df_reviews = pd.read_csv("data/sentiment_reviews.csv")
    append_sentiment_score(df, df_reviews)

    df, vectorizer = tfidf_amenities(df, max_features=100)

    return df, vectorizer

def get_amenities_suggestions():
    pass