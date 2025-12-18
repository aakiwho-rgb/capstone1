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

export interface PredictionResponse {
    smiles: string;
    prediction: "Active" | "Inactive";
    confidence: number;
    probability: number;
    lipinski: LipinskiStats;
    descriptors?: ExtendedDescriptors;
    status: string;
    timestamp: string;
}

export interface MoleculeSVGResponse {
    smiles: string;
    svg: string;
    status: string;
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
