import pandas as pd

# Load your data
data = pd.read_csv("data/listings/listings.csv")

# Apply custom styling for scrollable table
styled_table = data.style.set_table_attributes('style="width: 100%; height: 400px; overflow: auto; display: block;"')

# Save the styled table as an HTML file
styled_table.to_html("table_output.html")

print("Table saved as table_output.html. Open it in your browser.")
