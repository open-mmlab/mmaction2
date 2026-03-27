import pickle
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('data/skeleton/ntu60_2d/', exist_ok=True)

# Load the data
with open('/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/processed_data/ntu60_2d.pkl', 'rb') as f:
    data = pickle.load(f)

# First, let's examine the structure of the data
print("Data type:", type(data))

# If data is a dictionary
if isinstance(data, dict):
    # Get the keys and split them
    keys = list(data.keys())
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(keys)
    train_keys = keys[:int(0.8 * len(keys))]
    val_keys = keys[int(0.8 * len(keys)):]

    # Create train and val datasets
    train_data = {k: data[k] for k in train_keys}
    val_data = {k: data[k] for k in val_keys}

    print(
        f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")
# If data is a list or another iterable
elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
    # Use keys from data if it's not a simple list
    if hasattr(data, 'keys'):
        items = list(data.keys())
    else:
        items = list(range(len(data)))

    np.random.seed(42)  # For reproducibility
    np.random.shuffle(items)
    train_items = items[:int(0.8 * len(items))]
    val_items = items[int(0.8 * len(items)):]

    # Create train and val datasets based on structure
    if hasattr(data, 'keys'):
        train_data = {k: data[k] for k in train_items}
        val_data = {k: data[k] for k in val_items}
    else:
        train_data = [data[i] for i in train_items]
        val_data = [data[i] for i in val_items]

    print(
        f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")
else:
    # If data has a different structure, print it for debugging
    print("Unexpected data structure. First few elements:")
    print(data)
    exit(1)

# Save the splits
with open('data/skeleton/ntu60_2d/ntu60_2d_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    print("Saved training data to data/skeleton/ntu60_2d/ntu60_2d_train.pkl")

with open('data/skeleton/ntu60_2d/ntu60_2d_val.pkl', 'wb') as f:
    pickle.dump(val_data, f)
    print("Saved validation data to data/skeleton/ntu60_2d/ntu60_2d_val.pkl")
