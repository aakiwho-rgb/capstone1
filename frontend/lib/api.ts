import {
    PredictionResponse,
    HealthResponse,
    TrainingResponse,
    ModelInfoResponse,
    ErrorResponse,
    MoleculeSVGResponse,
    NameToSmilesResponse
} from "@/types/api";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL && process.env.NEXT_PUBLIC_API_URL.length > 0)
    ? process.env.NEXT_PUBLIC_API_URL
    : "http://127.0.0.1:8000";

class ApiError extends Error {
    constructor(public message: string, public status: number, public detail?: string) {
        super(message);
        this.name = "ApiError";
    }
}

async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
        let errorMessage = "An unexpected error occurred";
        let errorDetail = "";

        try {
            const errorData: any = await response.json(); // Use any to separate parsing from typing

            if (errorData.detail) {
                if (typeof errorData.detail === 'string') {
                    errorMessage = errorData.detail;
                } else if (Array.isArray(errorData.detail)) {
                    // Handle Pydantic validation errors
                    errorMessage = errorData.detail
                        .map((err: any) => err.msg || "Validation error")
                        .join(", ");
                } else if (typeof errorData.detail === 'object') {
                    errorMessage = JSON.stringify(errorData.detail);
                }
            }

            errorDetail = errorData.error_type || "";
        } catch {
            // Fallback if JSON parsing fails
            errorMessage = response.statusText;
        }

        if (response.status === 400) {
            throw new ApiError(errorMessage, 400, errorDetail);
        } else if (response.status === 503) {
            throw new ApiError(errorMessage, 503, errorDetail);
        }

        throw new ApiError(errorMessage, response.status, errorDetail);
    }

    return response.json();
}

export const api = {
    /**
     * Predict bioactivity for a given SMILES string
     */
    predict: async (smiles: string): Promise<PredictionResponse> => {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ smiles }),
        });
        return handleResponse<PredictionResponse>(response);
    },

    /**
     * Check system health and model status
     */
    checkHealth: async (): Promise<HealthResponse> => {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            return handleResponse<HealthResponse>(response);
        } catch (error) {
            // Handle network errors (offline backend)
            throw new ApiError("System unreachable", 0);
        }
    },

    /**
     * Trigger model retraining
     */
    train: async (): Promise<TrainingResponse> => {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: "POST",
        });
        return handleResponse<TrainingResponse>(response);
    },

    /**
     * Get detailed model configuration and metrics
     */
    getModelInfo: async (): Promise<ModelInfoResponse> => {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        return handleResponse<ModelInfoResponse>(response);
    },

    /**
     * Generate SVG representation of a molecule from SMILES
     */
    getMoleculeSVG: async (smiles: string, width = 300, height = 300): Promise<MoleculeSVGResponse> => {
        const response = await fetch(`${API_BASE_URL}/molecule/svg`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ smiles, width, height }),
        });
        return handleResponse<MoleculeSVGResponse>(response);
    },

    /**
     * Convert a molecule name to SMILES using PubChem API
     */
    nameToSmiles: async (name: string): Promise<NameToSmilesResponse> => {
        const response = await fetch(`${API_BASE_URL}/molecule/name-to-smiles`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ name }),
        });
        return handleResponse<NameToSmilesResponse>(response);
    }
};
