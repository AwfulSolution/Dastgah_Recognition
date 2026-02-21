from typing import Literal

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(
    model_type: Literal["lr", "svm"],
    seed: int,
    use_pca: bool,
    pca_variance: float,
) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))

    if model_type == "lr":
        clf = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            random_state=seed,
        )
    elif model_type == "svm":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    steps.append(("clf", clf))
    return Pipeline(steps)
