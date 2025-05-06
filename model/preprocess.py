import pandas as pd

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

def embedd_amenity(amenities: list[str], embedding_model):
    return embedding_model.encode(amenities)

def itemize_amenities(embedding_model, listings_df: pd.DataFrame):
    """
    Converts the 'amenities' column of stringified lists into multi-hot encoded columns.
    Modifies listings_df in place by appending amenity columns.
    """

    listings_df['embedded_amenities'] = listings_df['amenities'].apply(embedd_amenity, args=(embedding_model,))

def load_and_preprocess(embedding_model):
    data = pd.read_csv("data/listings/listings.csv")
    important_columns = [
        'id', 'price', 'neighbourhood_cleansed', 'room_type',
        'bedrooms', 'bathrooms', 'accommodates', 'amenities',
        'minimum_nights', 'number_of_reviews', 'review_scores_rating',
        'name', 'description'
    ]
    df = data[important_columns].copy()
    df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)
    df = df.dropna(subset=['price'])
    df = df[df['price'] != 0]

    df_reviews = pd.read_csv("data/sentiment_reviews.csv")
    append_sentiment_score(df, df_reviews)
    itemize_amenities(embedding_model, df)
    return df
