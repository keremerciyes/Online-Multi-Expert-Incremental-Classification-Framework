from motor_anomaly_classification.models.factory import OptimizerConfig
from motor_anomaly_classification.systems.base_classification import BaseClassificationModule, ModelConfig


class OfflineClassificationModule(BaseClassificationModule):
    def __init__(self, model_config: ModelConfig, optimizer_config: OptimizerConfig):
        super().__init__(model_config=model_config, optimizer_config=optimizer_config)
