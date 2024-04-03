import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from CSV
data = pd.read_csv('data/news10000.csv')

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Split the train data into train and validate sets
train_data, validate_data = train_test_split(train_data, test_size=0.25, random_state=42)

# Save the split datasets to separate CSV files
train_data.to_csv('data/train.csv', index=False)
validate_data.to_csv('data/validate.csv', index=False)
test_data.to_csv('data/test.csv', index=False)