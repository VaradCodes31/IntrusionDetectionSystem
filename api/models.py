from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class PacketData(BaseModel):
    # Flexible container for packet features
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    consensus_locked: bool
    latency_ms: float
    engine: str

class Alert(BaseModel):
    timestamp: str
    message: str
    level: str

class ExplanationRequest(BaseModel):
    packet_index: int
    data_source: str = "simulation" # simulation or live

class FeatureImportance(BaseModel):
    feature_names: List[str]
    importance_values: List[float]

class ExplanationResponse(BaseModel):
    label: str
    shap_plot_base64: Optional[str] = None
    lime_chart_data: Optional[FeatureImportance] = None
    anchor_rule: Optional[str] = None
    counterfactuals: Optional[List[Dict[str, Any]]] = None
