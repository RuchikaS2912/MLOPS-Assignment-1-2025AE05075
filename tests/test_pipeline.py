import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.data import FEATURES
from src.train import make_pipeline


def test_pipeline_fits_with_missing_values():
    rows = []
    for i in range(30):
        rows.append(
            [
                40 + i,
                i % 2,
                1 + (i % 4),
                120 + i,
                200 + i,
                i % 2,
                i % 3,
                140 + i,
                i % 2,
                float(i % 3),
                1 + i % 3,
                i % 3,
                3 + i % 2,
            ]
        )
    X = pd.DataFrame(rows, columns=FEATURES)
    X.loc[0, "chol"] = float("nan")
    y = [i % 2 for i in range(30)]
    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == 30
