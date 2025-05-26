import pandas as pd
from sklearn.model_selection import train_test_split

def append_sentiment_score(listings_df: pd.DataFrame, reviews_df: pd.DataFrame):
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

    upper = df['price'].quantile(0.85)
    df = df[(df['price'] <= upper)]

    df_reviews = pd.read_csv("data/sentiment_reviews.csv")
    append_sentiment_score(df, df_reviews)

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    # Save CSVs (optional)
    df.to_csv("data/listings/df_listings.csv", index=False)
    df_train.to_csv("data/listings/df_listings_train.csv", index=False)
    df_test.to_csv("data/listings/df_listings_test.csv", index=False)

    return df_train, df_test

def get_amenities_suggestions():
    pass