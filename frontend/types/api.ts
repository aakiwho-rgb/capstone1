export interface LipinskiStats {
    MW: number;
    LogP: number;
    NumHDonors: number;
    NumHAcceptors: number;
}

export interface ExtendedDescriptors {
    // Lipinski core
    MW: number;
    LogP: number;
    NumHDonors: number;
    NumHAcceptors: number;
    // Extended properties
    TPSA: number;
    NumRotatableBonds: number;
    NumAromaticRings: number;
    FractionCSP3: number;
    NumHeavyAtoms: number;
    MolecularFormula: string;
}

export interface ADMETDescriptors {
    CacoPermeability: string;
    HumanIntestinalAbsorption: string;
    BBBPermeant: string;
    PlasmaProteinBinding: string;
    hERG_Inhibition: string;
}

export interface DrugLikenessScores {
    qed: number;  // Quantitative Estimate of Drug-likeness (0-1)
    qed_category: "Excellent" | "Good" | "Moderate" | "Poor";
    lipinski_violations: number;  // 0-4
    lipinski_compliant: boolean;
    veber_compliant: boolean;
    pains_alerts: string[];  // List of PAINS filter matches
    pains_count: number;
}

export interface PredictionResponse {
    smiles: string;
    prediction: "Active" | "Inactive";
    confidence: number;
    probability: number;
    lipinski: LipinskiStats;
    descriptors?: ExtendedDescriptors;
    admet?: ADMETDescriptors;
    drug_likeness?: DrugLikenessScores;
    status: string;
    timestamp: string;
}

export interface MoleculeSVGResponse {
    smiles: string;
    svg: string;
    status: string;
}

export interface MoleculeImageExportResponse {
    smiles: string;
    format: "svg" | "png";
    content: string;  // base64 encoded
    filename: string;
    mime_type: string;
    width: number;
    height: number;
}

export interface NameToSmilesResponse {
    name: string;
    smiles: string;
    status: string;
}

export interface HealthResponse {
    status: string;
    model_ready: boolean;
    backend: string;
    version: string;
    model_info: Record<string, any>;
    timestamp: string;
}

export interface TrainingResponse {
    message: string;
    status: string;
    timestamp: string;
}

export interface ErrorResponse {
    detail: string;
    error_type: string;
    timestamp: string;
}

export interface ModelMetadata {
    training_date: string;
    training_samples: number;
    test_samples: number;
    metrics: {
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;
        confusion_matrix: number[][];
    };
    config: {
        fingerprint_radius: number;
        fingerprint_size: number;
        activity_threshold_nm: number;
    };
}

export interface ModelInfoResponse {
    status: string;
    metadata: ModelMetadata;
    configuration: {
        fingerprint_radius: number;
        fingerprint_size: number;
        activity_threshold_nm: number;
        confidence_threshold: number;
    };
}

export interface SimilarCompound {
    smiles: string;
    chembl_id: string;
    activity: "Active" | "Inactive";
    similarity: number;
    ic50_nm: number | null;
}

export interface SimilaritySearchResponse {
    query_smiles: string;
    results: SimilarCompound[];
    count: number;
}

export interface CalibrationData {
    predicted_probabilities: number[];
    true_probabilities: number[];
}

export interface FeatureImportance {
    indices: number[];
    importances: number[];
    top_k: number;
}

export interface DatasetStats {
    total_samples: number;
    active_count: number;
    inactive_count: number;
    ic50_distribution: {
        min: number;
        max: number;
        median: number;
        mean: number;
        q25: number;
        q75: number;
    };
    mw_stats: {
        min: number;
        max: number;
        mean: number;
        median: number;
    };
    logp_stats: {
        min: number;
        max: number;
        mean: number;
        median: number;
    };
}

export interface PredictionHistoryEntry {
    smiles: string;
    prediction: "Active" | "Inactive";
    confidence: number;
    probability: number;
    timestamp: string;
}

export interface HistoryResponse {
    predictions: PredictionHistoryEntry[];
    count: number;
    available: boolean;
}

export interface PDFExportResponse {
    pdf: string; // base64
    filename: string;
    mime_type: string;
}
