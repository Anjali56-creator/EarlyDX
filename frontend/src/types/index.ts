export interface DiseaseInfo {
  key: string;
  display_name: string;
  group: string;
  model_available: boolean;
  unavailable_reason: string | null;
  required_features: string[];
}

export interface FeatureDetail {
  type: string;
  unit: string;
  min: number;
  max: number;
  required: boolean;
}

export interface FeatureSchema {
  disease: string;
  model_id: string;
  target: string;
  required_features: string[];
  feature_details: Record<string, FeatureDetail>;
}

export interface ImportantFeature {
  feature: string;
  value: number;
  population_median: number | null;
  direction: string;
  impact: string;
  importance: number;
}

export interface PredictionResponse {
  disease: string;
  disease_key: string;
  risk_score: number;
  risk_level: "LOW" | "MODERATE" | "HIGH";
  calibrated: boolean;
  model_version: string;
  model_algorithm: string | null;
  model_performance: Record<string, number | null>;
  important_features: ImportantFeature[];
  threshold_policy: string | null;
  disclaimer: string;
  session_id: number | null;
}

export interface PredictAllResponse {
  results: PredictionResponse[];
  skipped: { disease: string; disease_key: string; reason: string; missing_features: string[] }[];
  disclaimer: string;
  session_id: number | null;
}

export interface ModelEntry {
  model_id: string;
  disease: string;
  version: string;
  algorithm: string;
  dataset: string;
  training_date: string;
  metrics: Record<string, number>;
  calibration: string;
  status: string;
  loaded: boolean;
}

export interface DatasetEntry {
  disease: string;
  dataset_name: string;
  source: string;
  source_url: string;
  license: string;
  records: number;
  target: string;
  status: string;
  limitations: string;
}

export interface Health {
  status: string;
  models_loaded: number;
  model_load_errors: Record<string, string>;
  database: { available: boolean; error: string | null };
}
