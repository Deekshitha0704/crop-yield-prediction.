import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

DATA_PATH = os.path.join(os.path.dirname(__file__), 'crop_yield.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_yield_model.pkl')


def make_sample_data(path):
    # create a tiny synthetic dataset so the script can run even if no CSV provided
    df = pd.DataFrame({
        'Rainfall': np.random.uniform(20, 200, size=50),
        'Temp': np.random.uniform(10, 35, size=50),
        'Area': np.random.uniform(0.5, 5.0, size=50),
        'Fertilizer': np.random.uniform(0, 100, size=50),
    })
    # simple synthetic target with noise
    df['Yield'] = (0.3 * df['Rainfall'] + 0.5 * df['Temp'] + 2.0 * df['Area'] + 0.1 * df['Fertilizer']
                   + np.random.normal(0, 5, size=len(df)))
    df.to_csv(path, index=False)
    return df


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}. Creating synthetic sample dataset.")
        data = make_sample_data(DATA_PATH)
    else:
        data = pd.read_csv(DATA_PATH)

    required_cols = ['Rainfall', 'Temp', 'Area', 'Fertilizer', 'Yield']
    if not all(c in data.columns for c in required_cols):
        raise ValueError(f"Input data must contain columns: {required_cols}")

    X = data[['Rainfall', 'Temp', 'Area', 'Fertilizer']]
    y = data['Yield']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Trained LinearRegression model and saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
