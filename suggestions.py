# review_insights.py

import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def get_keywords_by_location_and_type(
    df_reviews: pd.DataFrame,
    df_listings: pd.DataFrame,
    neighborhood: str,
    room_type: str,
    top_n: int = 20
) -> list[tuple[str, int]]:
    """
    Extracts the top positive review keywords for a specific neighborhood and room type.

    Parameters:
        df_reviews: DataFrame with sentiment-labeled reviews (must include 'listing_id', 'sentiment', 'comments')
        df_listings: DataFrame with listing metadata (must include 'id', 'neighbourhood_cleansed', 'room_type')
        neighborhood: Selected neighborhood (e.g. 'Södermalm')
        room_type: Selected room type (e.g. 'Private room')
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, count) tuples
    """

    # Merge reviews with listing metadata
    merged = df_reviews.merge(
        df_listings[['id', 'neighbourhood_cleansed', 'room_type']],
        left_on='listing_id',
        right_on='id',
        how='inner'
    )

    # Filter to positive reviews in selected context
    filtered = merged[
        (merged['sentiment'] == 'POSITIVE') &
        (merged['neighbourhood_cleansed'] == neighborhood) &
        (merged['room_type'] == room_type)
    ]

    if filtered.empty:
        return []

    # Tokenize and count words
    comments = filtered['comments'].dropna().str.lower().tolist()
    all_text = ' '.join(comments)
    tokens = re.findall(r'\b[a-z]{3,}\b', all_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return Counter(filtered_tokens).most_common(top_n)
