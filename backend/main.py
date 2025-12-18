import os
import sys
import logging
import joblib
import threading
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# Third-party imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors, Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized application configuration"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_FILE = BASE_DIR / "data" / "CHEMBL1824_bioactivity.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_FILE = MODEL_DIR / "her2_rf_model.joblib"
    METADATA_FILE = MODEL_DIR / "model_metadata.joblib"
    
    # Data Parameters
    ALLOWED_UNITS = ['nM', 'uM']
    ACTIVITY_THRESHOLD_NM = 1000.0  # 1 uM
    
    # Model Parameters
    FINGERPRINT_RADIUS = 2
    FINGERPRINT_SIZE = 2048
    CONFIDENCE_THRESHOLD = 0.5
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # API Settings
    API_TITLE = "HER2 Drug Discovery Platform"
    API_VERSION = "3.0.0"
    
    @classmethod
    def ensure_directories(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Initialize
Config.ensure_directories()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HER2_API")

# ============================================================================
# PYDANTIC MODELS (MATCHING FRONTEND)
# ============================================================================

class LipinskiStats(BaseModel):
    MW: float
    LogP: float
    NumHDonors: int
    NumHAcceptors: int

class ExtendedDescriptors(BaseModel):
    """Extended molecular descriptors for visualization"""
    # Lipinski core
    MW: float
    LogP: float
    NumHDonors: int
    NumHAcceptors: int
    # Extended properties
    TPSA: float  # Topological Polar Surface Area
    NumRotatableBonds: int
    NumAromaticRings: int
    FractionCSP3: float  # Fraction of sp3 carbons
    NumHeavyAtoms: int
    MolecularFormula: str

class PredictionRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="SMILES string of the molecule")

class PredictionResponse(BaseModel):
    smiles: str
    prediction: str  # "Active" | "Inactive"
    confidence: float
    probability: float
    lipinski: LipinskiStats
    descriptors: Optional[ExtendedDescriptors] = None
    status: str = "success"
    timestamp: str

class MoleculeSVGRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="SMILES string of the molecule")
    width: int = Field(default=300, ge=100, le=800)
    height: int = Field(default=300, ge=100, le=800)

class MoleculeSVGResponse(BaseModel):
    smiles: str
    svg: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    version: str
    backend: str
    model_info: Optional[Dict[str, Any]] = None
    timestamp: str

class TrainingResponse(BaseModel):
    message: str
    status: str  # "started" | "already_running" | "error"
    timestamp: str

class ModelInfoResponse(BaseModel):
    status: str
    metadata: Dict[str, Any]
    configuration: Dict[str, Any]

class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    timestamp: str

# ============================================================================
# CORE LOGIC: DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Robust data cleaning and preprocessing pipeline"""
    
    @staticmethod
    def load_and_clean_data(filepath: Path) -> pd.DataFrame:
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}")
            
        logger.info(f"Loading dataset from {filepath}...")
        try:
            # Use semi-colon separator as seen in dataset
            df = pd.read_csv(filepath, sep=';')
            
            # Normalize headers: strip whitespace, check required columns
            df.columns = df.columns.str.strip()
            
            # Map columns to internal names
            col_map = {
                'Molecule ChEMBL ID': 'chembl_id',
                'Smiles': 'smiles',
                'Standard Value': 'value',
                'Standard Units': 'units',
                'Standard Relation': 'relation'
            }
            
            missing = [c for c in col_map.keys() if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            df = df.rename(columns=col_map)
            df = df[list(col_map.values())].copy()
            
            # 1. Clean Units (Convert uM -> nM)
            df = df[df['units'].isin(Config.ALLOWED_UNITS)].copy()
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Conversion mask
            mask_um = df['units'] == 'uM'
            df.loc[mask_um, 'value'] = df.loc[mask_um, 'value'] * 1000.0
            df['units'] = 'nM'  # Now all are nM
            
            # 2. Clean Relations and Determine Class
            # Clean relation strings
            df['relation'] = df['relation'].astype(str).str.replace("'", "").str.strip()
            
            def label_compound(row):
                val = row['value']
                rel = row['relation']
                thresh = Config.ACTIVITY_THRESHOLD_NM
                
                if pd.isna(val): return None
                
                # Definitive Case: '=' or '~' (approx)
                if rel in ['=', '~', 'nan', 'None']:
                    return 1 if val <= thresh else 0
                
                # Bound Cases
                # < 500 (Active) vs < 5000 (Ambiguous)
                if rel in ['<', '<=']:
                    return 1 if val <= thresh else None
                    
                # > 5000 (Inactive) vs > 500 (Ambiguous)
                if rel in ['>', '>=']:
                    return 0 if val > thresh else None
                    
                return None

            df['activity'] = df.apply(label_compound, axis=1)
            df = df.dropna(subset=['activity', 'smiles'])
            
            # 3. Handle Duplicates (Aggregation)
            # If a SMILES appears twice, take the majority vote for activity 
            # or the geometric mean of values (more accurate), but here we have binary labels.
            # We will group by SMILES and take the mode (most frequent label)
            # Logic: If mixed signals, drop it to be safe (high quality data only).
            
            # Check for consistency
            consistency = df.groupby('smiles')['activity'].mean()
            # Keep only purely Active (1.0) or purely Inactive (0.0) groups
            # 0.5 means one active one inactive record -> ambiguous -> drop
            valid_smiles = consistency[consistency.isin([0.0, 1.0])].index
            
            df_clean = df[df['smiles'].isin(valid_smiles)].drop_duplicates(subset=['smiles'])
            
            logger.info(f"Data Cleaning Complete. Valid samples: {len(df_clean)}")
            logger.info(f"Class Balance: {df_clean['activity'].value_counts(normalize=True).to_dict()}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise

# ============================================================================
# CORE LOGIC: MODEL MANAGER
# ============================================================================

class ModelManager:
    """Thread-safe Manager for the Machine Learning Model"""
    
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=Config.FINGERPRINT_RADIUS, 
            fpSize=Config.FINGERPRINT_SIZE
        )
        self.lock = threading.RLock()
        self.is_training = False

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load_model(self) -> bool:
        """Load model from disk safely"""
        with self.lock:
            if not Config.MODEL_FILE.exists():
                return False
            try:
                self.model = joblib.load(Config.MODEL_FILE)
                if Config.METADATA_FILE.exists():
                    self.metadata = joblib.load(Config.METADATA_FILE)
                logger.info("✅ Model loaded successfully from disk")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

    def train_model(self):
        """Execute training pipeline (meant for background task)"""
        with self.lock:
            if self.is_training:
                logger.warning("Training already in progress")
                return
            self.is_training = True

        try:
            logger.info("🚀 Starting training pipeline...")
            start_time = datetime.now()
            
            # 1. Load Data
            df = DataProcessor.load_and_clean_data(Config.DATA_FILE)
            
            # 2. Feature Generation
            smiles_list = df['smiles'].tolist()
            y = df['activity'].astype(int).values
            X = []
            
            valid_indices = []
            for i, s in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(s)
                if mol:
                    fp = self.mfpgen.GetFingerprint(mol)
                    X.append(np.array(fp))
                    valid_indices.append(i)
            
            X = np.array(X)
            y = y[valid_indices]
            
            if len(y) < 50:
                raise ValueError("Insufficient data for training (<50 samples)")

            # 3. Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
            )
            
            # 4. Train (Random Forest)
            # Random Forest is often more robust than MLP for raw fingerprints
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=Config.RANDOM_STATE,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)
            
            # 5. Evaluate
            y_pred = clf.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "training_samples": len(X_train)
            }
            
            logger.info(f"Training Results: {metrics}")
            
            # 6. Save
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "config": {
                    "threshold_nm": Config.ACTIVITY_THRESHOLD_NM,
                    "model_type": "RandomForestClassifier"
                }
            }
            
            with self.lock:
                joblib.dump(clf, Config.MODEL_FILE)
                joblib.dump(metadata, Config.METADATA_FILE)
                self.model = clf
                self.metadata = metadata
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Training finished in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
        finally:
            self.is_training = False

    def predict(self, smiles: str) -> PredictionResponse:
        """Predict bioactivity for a single SMILES string"""
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Model is not ready")
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
            
        # Core Lipinski Descriptors
        lipinski = LipinskiStats(
            MW=round(Descriptors.MolWt(mol), 2),
            LogP=round(Descriptors.MolLogP(mol), 3),
            NumHDonors=Lipinski.NumHDonors(mol),
            NumHAcceptors=Lipinski.NumHAcceptors(mol)
        )
        
        # Extended Descriptors for visualization
        extended = ExtendedDescriptors(
            MW=lipinski.MW,
            LogP=lipinski.LogP,
            NumHDonors=lipinski.NumHDonors,
            NumHAcceptors=lipinski.NumHAcceptors,
            TPSA=round(Descriptors.TPSA(mol), 2),
            NumRotatableBonds=Lipinski.NumRotatableBonds(mol),
            NumAromaticRings=rdMolDescriptors.CalcNumAromaticRings(mol),
            FractionCSP3=round(Lipinski.FractionCSP3(mol), 3),
            NumHeavyAtoms=Lipinski.HeavyAtomCount(mol),
            MolecularFormula=CalcMolFormula(mol)
        )
        
        # Fingerprint
        fp = self.mfpgen.GetFingerprint(mol)
        fp_arr = np.array(fp).reshape(1, -1)
        
        # Predict
        # Classes: 0=Inactive, 1=Active
        probs = self.model.predict_proba(fp_arr)[0]
        active_prob = probs[1]
        
        is_active = active_prob >= Config.CONFIDENCE_THRESHOLD
        label = "Active" if is_active else "Inactive"
        confidence = active_prob if is_active else (1 - active_prob)
        
        return PredictionResponse(
            smiles=smiles,
            prediction=label,
            confidence=round(confidence, 4),
            probability=round(active_prob, 4),
            lipinski=lipinski,
            descriptors=extended,
            status="success",
            timestamp=datetime.now().isoformat()
        )

# Global Instance
model_manager = ModelManager()

# ============================================================================
# FASTAPI APP LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up HER2 Platform...")
    
    # Try loading existing model
    loaded = model_manager.load_model()
    
    if not loaded:
        logger.warning("No model found. Triggering initial training...")
        # Run training in a separate thread so startup doesn't hang
        train_thread = threading.Thread(target=model_manager.train_model)
        train_thread.start()
    
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_type="client_error" if exc.status_code < 500 else "server_error",
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="An internal server error occurred.",
            error_type="internal_server_error",
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health and model status"""
    return HealthResponse(
        status="online",
        model_ready=model_manager.is_ready,
        version=Config.API_VERSION,
        backend="RandomForest (scikit-learn)",
        model_info=model_manager.metadata.get("metrics"),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    """Predict bioactivity for a molecule"""
    return model_manager.predict(payload.smiles)

@app.post("/train", response_model=TrainingResponse)
async def trigger_training(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    if model_manager.is_training:
        return TrainingResponse(
            message="Training is already in progress",
            status="already_running",
            timestamp=datetime.now().isoformat()
        )
    
    background_tasks.add_task(model_manager.train_model)
    return TrainingResponse(
        message="Training initiated in background",
        status="started",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model configuration"""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return ModelInfoResponse(
        status="ready",
        metadata=model_manager.metadata,
        configuration={
            "radius": Config.FINGERPRINT_RADIUS,
            "bits": Config.FINGERPRINT_SIZE,
            "threshold_nm": Config.ACTIVITY_THRESHOLD_NM
        }
    )

@app.post("/molecule/svg", response_model=MoleculeSVGResponse)
async def generate_molecule_svg(payload: MoleculeSVGRequest):
    """Generate 2D SVG representation of a molecule from SMILES"""
    mol = Chem.MolFromSmiles(payload.smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        # Generate 2D coordinates and SVG
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
        
        drawer = Draw.MolDraw2DSVG(payload.width, payload.height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        return MoleculeSVGResponse(
            smiles=payload.smiles,
            svg=svg,
            status="success"
        )
    except Exception as e:
        logger.error(f"SVG generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate molecule SVG")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)