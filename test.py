import pandas as pd

# Load your data
data = pd.read_csv("data/listings/df_listings.csv")
data_t = pd.read_csv("data/listings/df_listings_train.csv")
data_te = pd.read_csv("data/listings/df_listings_test.csv")

print(data.shape)
print(data_t.shape)
print(data_te.shape)