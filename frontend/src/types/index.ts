export interface DiseaseInfo {
  key: string;
  display_name: string;
  group: string;
  model_available: boolean;
  unavailable_reason: string | null;
  required_features: string[];
}

export interface FeatureDetail {
  type: "number" | "categorical";
  unit: string;
  min?: number;
  max?: number;
  options?: string[];
  required: boolean;
  /** If true, an input of exactly 0 means "not measured": the model's own
   * preprocessing treats 0 as missing and imputes it, so 0 is valid input
   * even though it falls outside [min, max]. */
  zero_is_missing?: boolean;
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
  /** The validation-derived cut points behind risk_level (real numbers from
   * this model's own training run, not invented clinical zones). */
  risk_thresholds?: { low_cut: number | null; high_cut: number | null } | null;
  /** The 0.5 cut used for this model's standard evaluation metrics
   * (accuracy/precision/recall/F1) — not a clinical threshold. */
  decision_threshold?: number | null;
  /** risk_score >= decision_threshold, as 0/1. */
  predicted_class?: number | null;
  disclaimer: string;
  session_id: number | null;
}

export interface ValidationSample {
  features: Record<string, number | string>;
  actual_outcome: 0 | 1;
  predicted_probability: number;
  predicted_outcome: 0 | 1;
  predicted_risk_level: "LOW" | "MODERATE" | "HIGH";
}

export interface ValidationSamples {
  disease: string;
  disease_key: string;
  model_version: string;
  note: string;
  samples: ValidationSample[];
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
  features?: string[];
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
  features?: string[];
  target: string;
  class_distribution?: Record<string, number>;
  preprocessing?: string;
  status: string;
  limitations: string;
}

/** Standard binary-classification metrics at a fixed threshold, exactly as
 * written to model_metadata.json by the training run. */
export interface ClassificationMetrics {
  n: number | null;
  threshold: number | null;
  accuracy: number | null;
  precision: number | null;
  recall_sensitivity: number | null;
  specificity: number | null;
  f1: number | null;
  roc_auc: number | null;
  pr_auc: number | null;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number } | null;
}

export interface EvaluationSummary {
  disease: string;
  disease_key: string;
  model_version: string;
  selected_algorithm: string;
  baseline_algorithm: string | null;
  best_validation_candidate_by_roc_auc: string | null;
  candidates_compared: string[];
  test_metrics: Omit<ClassificationMetrics, "threshold" | "confusion_matrix">;
  calibration_method: string | null;
  cv_roc_auc: { mean: number; std: number } | null;
}

export interface EvaluationDetail {
  disease: string;
  disease_key: string;
  model_version: string;
  selected_algorithm: string;
  baseline_algorithm: string | null;
  selection_rule: string | null;
  best_validation_candidate_by_roc_auc: string | null;
  training_date: string | null;
  dataset: { name?: string; registry_key?: string };
  split: { scheme?: string; sizes?: { train: number; val: number; test: number } };
  target_positive_rate: number | null;
  candidates: Record<string, ClassificationMetrics>;
  test_metrics: ClassificationMetrics & {
    class_distribution: Record<string, number> | null;
    brier: number | null;
    calibration_bins:
      | { bin: [number, number]; count: number; mean_predicted: number; observed_frequency: number }[]
      | null;
  };
  test_metrics_at_screening_threshold: ClassificationMetrics | null;
  cross_validation: {
    folds: number | null;
    folds_requested?: number | null;
    summary: Record<string, { mean: number; std: number }> | null;
  };
  calibration: { method: string | null; validation_brier: Record<string, number> | null };
  risk_thresholds: {
    low_cut: number | null;
    high_cut: number | null;
    policy?: string;
    target_sensitivity?: number;
    target_specificity?: number;
  } | null;
  permutation_importance: { feature: string; importance: number; std: number }[] | null;
  unavailable: Record<string, string>;
  limitations: string | null;
  ethical_note: string | null;
  has_validation_samples: boolean;
}

export interface RecentAssessments {
  available: boolean;
  total_sessions: number;
  results: {
    session_id: number;
    disease_key: string;
    model_id: string;
    risk_score: number;
    risk_level: "LOW" | "MODERATE" | "HIGH";
    created_at: string | null;
  }[];
}

export interface Health {
  status: string;
  models_loaded: number;
  model_load_errors: Record<string, string>;
  database: { available: boolean; error: string | null };
}
