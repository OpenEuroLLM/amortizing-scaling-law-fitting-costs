import numpy as np

from .base import SurrogateModel


class AutoGluonModel(SurrogateModel):
    def __init__(self, seed: int | None = None, **kwargs):
        self.seed = seed

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        import pandas as pd
        from autogluon.tabular import TabularPredictor

        train_data = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
        train_data["labels"] = train_y.ravel()
        # AutoGluon has no single top-level seed knob; run-to-run reproducibility relies
        # on the global NumPy/torch seeding done in the pipeline (seed_everything).
        self._predictor = TabularPredictor(label="labels").fit(train_data)
        self._feature_names = train_data.columns.drop("labels").tolist()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        import pandas as pd

        X_df = pd.DataFrame(X_test, columns=self._feature_names)
        return self._predictor.predict(X_df).to_numpy()
