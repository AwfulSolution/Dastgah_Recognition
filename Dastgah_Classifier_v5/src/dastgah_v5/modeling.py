from typing import Literal

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_TYPES = (
    "lr",
    "svm",
    "svm_linear",
    "svm_rbf",
    "knn",
    "rf",
    "extratrees",
    "catboost",
    "ensemble",
)


def _sk_n_jobs(model_jobs: int) -> int:
    if model_jobs == 0:
        return -1
    return model_jobs


def _catboost_threads(model_jobs: int) -> int:
    if model_jobs <= 0:
        return -1
    return model_jobs


def _build_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )


def _build_svm_linear(seed: int) -> SVC:
    return SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )


def _build_svm_rbf(seed: int) -> SVC:
    return SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )


def _build_knn(model_jobs: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=9,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=_sk_n_jobs(model_jobs),
    )


def _build_rf(seed: int, model_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=_sk_n_jobs(model_jobs),
    )


def _build_extratrees(seed: int, model_jobs: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=_sk_n_jobs(model_jobs),
    )


def _build_catboost(seed: int, model_jobs: int):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "catboost is not installed. Install it with `pip install catboost` "
            "or use a different --model_type."
        ) from exc
    return CatBoostClassifier(
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        iterations=700,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        thread_count=_catboost_threads(model_jobs),
        allow_writing_files=False,
        verbose=False,
    )


def _build_classifier(model_type: str, seed: int, model_jobs: int):
    if model_type == "lr":
        return _build_lr(seed)
    if model_type == "svm_linear":
        return _build_svm_linear(seed)
    if model_type in ("svm", "svm_rbf"):
        return _build_svm_rbf(seed)
    if model_type == "knn":
        return _build_knn(model_jobs)
    if model_type == "rf":
        return _build_rf(seed, model_jobs)
    if model_type == "extratrees":
        return _build_extratrees(seed, model_jobs)
    if model_type == "catboost":
        return _build_catboost(seed, model_jobs)
    if model_type == "ensemble":
        # Keep VotingClassifier single-threaded and apply jobs at estimator level.
        base_jobs = model_jobs
        estimators = [
            ("svm_rbf", _build_svm_rbf(seed)),
            ("extratrees", _build_extratrees(seed, base_jobs)),
            ("lr", _build_lr(seed)),
        ]
        weights = [2, 2, 1]
        try:
            estimators.append(("catboost", _build_catboost(seed, base_jobs)))
            weights.append(2)
        except ImportError:
            pass
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def build_model(
    model_type: Literal["lr", "svm", "svm_linear", "svm_rbf", "knn", "rf", "extratrees", "catboost", "ensemble"],
    seed: int,
    use_pca: bool,
    pca_variance: float,
    model_jobs: int = 1,
) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))

    clf = _build_classifier(model_type=model_type, seed=seed, model_jobs=model_jobs)
    steps.append(("clf", clf))
    return Pipeline(steps)
