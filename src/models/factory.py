"""
Model factory for creating and instantiating models by name.
"""

from typing import Any, Dict, Optional

from src.models.base import BaseModel
from src.models.baselines.dcn import DCNModel
from src.models.baselines.ft_transformer import FTTransformerModel
from src.models.baselines.logistic_regression import LogisticRegressionModel
from src.models.baselines.xgboost_model import XGBoostModel
from src.models.mambatab import MambaTabDModel, MambaTabTModel

MODEL_REGISTRY = {
    "mambatab_d": MambaTabDModel,
    "mambatab": MambaTabDModel,
    "mambatab_t": MambaTabTModel,
    "xgboost": XGBoostModel,
    "xgb": XGBoostModel,
    "logistic_regression": LogisticRegressionModel,
    "lr": LogisticRegressionModel,
    "ft_transformer": FTTransformerModel,
    "ft_trans": FTTransformerModel,
    "dcn": DCNModel,
}

# Table VI from the paper: Tuned hyperparameters per dataset for MambaTab-T
MAMBATAB_T_CONFIGS = {
    "adult": {"embedding_dim": 64, "ssm_state_dim": 64, "block_expansion": 10},
    "ad": {"embedding_dim": 64, "ssm_state_dim": 64, "block_expansion": 10},
    "income": {"embedding_dim": 32, "ssm_state_dim": 64, "block_expansion": 1},
    "ic": {"embedding_dim": 32, "ssm_state_dim": 64, "block_expansion": 1},
    "blastchar": {"embedding_dim": 16, "ssm_state_dim": 4, "block_expansion": 10},
    "bl": {"embedding_dim": 16, "ssm_state_dim": 4, "block_expansion": 10},
    "credit_g": {"embedding_dim": 64, "ssm_state_dim": 16, "block_expansion": 3},
    "cg": {"embedding_dim": 64, "ssm_state_dim": 16, "block_expansion": 3},
    "cylinder_bands": {"embedding_dim": 32, "ssm_state_dim": 8, "block_expansion": 7},
    "cb": {"embedding_dim": 32, "ssm_state_dim": 8, "block_expansion": 7},
}


def create_model(
    model_name: str,
    num_features: int,
    config: Dict[str, Any],
    dataset_name: Optional[str] = None,
) -> BaseModel:
    """Create and return a model instance configured according to the config.

    Args:
        model_name: Identifier for the model.
        num_features: Number of input features for the dataset.
        config: Full configuration dictionary.

    Returns:
        Instantiated BaseModel.

    Raises:
        ValueError: If model_name is not registered.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[key]

    if key in ("mambatab_d", "mambatab"):
        cfg = config.get("mambatab", {})
        return model_cls(
            num_features=num_features,
            embedding_dim=cfg.get("embedding_dim", 32),
            ssm_state_dim=cfg.get("ssm_state_dim", 32),
            conv_width=cfg.get("conv_width", 4),
            num_blocks=cfg.get("num_blocks", 1),
            block_expansion=cfg.get("block_expansion", 2),
        )

    elif key == "mambatab_t":
        ds_key = dataset_name.lower().strip() if dataset_name else "adult"
        # Check YAML config first, then fallback to built-in Table VI defaults
        yaml_t_cfgs = config.get("mambatab_t", {})
        t_cfg = yaml_t_cfgs.get(ds_key, MAMBATAB_T_CONFIGS.get(ds_key, {}))
        return model_cls(
            num_features=num_features,
            embedding_dim=t_cfg.get("embedding_dim", 32),
            ssm_state_dim=t_cfg.get("ssm_state_dim", 32),
            conv_width=t_cfg.get("conv_width", 4),
            num_blocks=t_cfg.get("num_blocks", 1),
            block_expansion=t_cfg.get("block_expansion", 2),
        )

    elif key in ("xgboost", "xgb"):
        cfg = config.get("xgboost", {})
        return model_cls(
            num_features=num_features,
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.1),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            early_stopping_rounds=cfg.get("early_stopping_rounds", 10),
        )

    elif key in ("logistic_regression", "lr"):
        cfg = config.get("logistic_regression", {})
        return model_cls(
            num_features=num_features,
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
        )

    elif key in ("ft_transformer", "ft_trans"):
        cfg = config.get("ft_transformer", {})
        return model_cls(
            num_features=num_features,
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            d_ffn=cfg.get("d_ffn", 128),
            dropout=cfg.get("dropout", 0.1),
            attention_dropout=cfg.get("attention_dropout", 0.1),
        )

    elif key == "dcn":
        cfg = config.get("dcn", {})
        return model_cls(
            num_features=num_features,
            hidden_dims=cfg.get("hidden_dims", [128, 64]),
            num_cross_layers=cfg.get("num_cross_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )

    raise ValueError(f"Unhandled model creation branch for '{model_name}'")
