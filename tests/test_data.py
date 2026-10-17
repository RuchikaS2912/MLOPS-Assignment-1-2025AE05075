import pandas as pd
from src.data import load_dataset, FEATURES


def test_target_is_binary_and_features_exist(tmp_path):
    cols = FEATURES + ["target"]
    row = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 2]
    path = tmp_path / "toy.csv"
    pd.DataFrame([row, row[:-1] + [0]], columns=cols).to_csv(path, index=False)
    df = pd.read_csv(path)
    assert set(df.target.unique()) == {0, 2}
    assert len(FEATURES) == 13


def test_load_dataset_normalizes_target(tmp_path):
    rows = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 2], [55, 0, 1, 130, 250, 0, 1, 160, 0, 1.2, 1, 0, 3, 0]]
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)
    result = load_dataset(str(path))
    assert set(result.target) == {0, 1}
