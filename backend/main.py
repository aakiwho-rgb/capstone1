import os
import sys
import logging
import joblib
import threading
import numpy as np
import pandas as pd
import httpx
import json
import io
import base64
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
import redis

# Third-party imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors, Draw, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.QED import qed
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
from rdkit.DataStructs import TanimotoSimilarity

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
    CALIBRATION_FILE = MODEL_DIR / "calibration_data.joblib"
    FEATURE_IMPORTANCE_FILE = MODEL_DIR / "feature_importance.joblib"
    DATASET_STATS_FILE = MODEL_DIR / "dataset_stats.joblib"
    DATASET_CACHE_FILE = MODEL_DIR / "dataset_cache.joblib"
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_CACHE_TTL = 86400  # 24 hours
    REDIS_HISTORY_KEY = "her2:history"
    REDIS_HISTORY_MAX = 1000  # Max history entries
    
    # Data Parameters
    ALLOWED_UNITS = ['nM', 'uM']
    ACTIVITY_THRESHOLD_NM = 1000.0  # 1 uM
    
    # Model Parameters
    FINGERPRINT_RADIUS = 2
    FINGERPRINT_SIZE = 2048
    CONFIDENCE_THRESHOLD = 0.5
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Similarity Search
    SIMILARITY_TOP_K = 10
    SIMILARITY_THRESHOLD = 0.7
    
    # API Settings
    API_TITLE = "HER2 Drug Discovery Platform"
    API_VERSION = "4.0.0"
    
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

class ADMETDescriptors(BaseModel):
    """ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) predictions"""
    CacoPermeability: str  # "High" | "Low"
    HumanIntestinalAbsorption: str  # "Good" | "Poor"
    BBBPermeant: str  # "Yes" | "No"
    PlasmaProteinBinding: str  # "High" | "Low"
    hERG_Inhibition: str  # "High Risk" | "Low Risk"

class DrugLikenessScores(BaseModel):
    """Drug-likeness scoring including QED, Lipinski, Veber, and PAINS"""
    qed: float  # Quantitative Estimate of Drug-likeness (0-1, higher = more drug-like)
    qed_category: str  # "Excellent" | "Good" | "Moderate" | "Poor"
    lipinski_violations: int  # Number of Lipinski Rule of 5 violations (0-4)
    lipinski_compliant: bool  # True if violations <= 1
    veber_compliant: bool  # True if TPSA <= 140 AND RotatableBonds <= 10
    pains_alerts: List[str]  # List of PAINS filter matches (empty = no alerts)
    pains_count: int  # Number of PAINS alerts

class PredictionRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="SMILES string of the molecule")

class PredictionResponse(BaseModel):
    smiles: str
    prediction: str  # "Active" | "Inactive"
    confidence: float
    probability: float
    lipinski: LipinskiStats
    descriptors: Optional[ExtendedDescriptors] = None
    admet: Optional[ADMETDescriptors] = None
    drug_likeness: Optional[DrugLikenessScores] = None
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

class NameToSmilesRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Molecule name to convert to SMILES")

class NameToSmilesResponse(BaseModel):
    name: str
    smiles: str
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

class MoleculeImageExportRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="SMILES string of the molecule")
    width: int = Field(default=400, ge=100, le=2000, description="Image width in pixels")
    height: int = Field(default=400, ge=100, le=2000, description="Image height in pixels")
    background: str = Field(default="white", description="Background color: 'white', 'transparent', or hex color")

class MoleculeImageExportResponse(BaseModel):
    smiles: str
    format: str  # "svg" | "png"
    content: str  # base64 encoded
    filename: str
    mime_type: str
    width: int
    height: int

class SimilarCompound(BaseModel):
    smiles: str
    chembl_id: Optional[str] = None
    activity: str
    similarity: float
    ic50_nm: Optional[float] = None

class SimilaritySearchRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class SimilaritySearchResponse(BaseModel):
    query_smiles: str
    similar_compounds: List[SimilarCompound]
    count: int
    timestamp: str

class CalibrationData(BaseModel):
    prob_pred: List[float]
    prob_true: List[float]
    sample_counts: List[int]

class FeatureImportance(BaseModel):
    feature_indices: List[int]
    importance_scores: List[float]
    top_k: int

class DatasetStats(BaseModel):
    total_samples: int
    active_count: int
    inactive_count: int
    class_balance: Dict[str, float]
    ic50_distribution: Dict[str, float]
    molecular_weight_stats: Dict[str, float]
    lipophilicity_stats: Dict[str, float]

class PredictionHistoryEntry(BaseModel):
    smiles: str
    prediction: str
    confidence: float
    probability: float
    timestamp: str
    molecule_name: Optional[str] = None

class HistoryResponse(BaseModel):
    history: List[PredictionHistoryEntry]
    count: int
    timestamp: str

# ============================================================================
# CORE LOGIC: REDIS MANAGER
# ============================================================================

class RedisManager:
    """Thread-safe Redis connection manager with caching support"""
    
    def __init__(self):
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.client.ping()
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. History will not persist.")
            self.client = None
    
    @property
    def is_available(self) -> bool:
        return self.client is not None
    
    def add_to_history(self, entry: PredictionHistoryEntry) -> bool:
        """Add prediction to history"""
        if not self.is_available:
            return False
        try:
            history_data = entry.model_dump_json()
            self.client.lpush(Config.REDIS_HISTORY_KEY, history_data)
            self.client.ltrim(Config.REDIS_HISTORY_KEY, 0, Config.REDIS_HISTORY_MAX - 1)
            return True
        except Exception as e:
            logger.error(f"Failed to add to history: {e}")
            return False
    
    def get_history(self, limit: int = 50) -> List[PredictionHistoryEntry]:
        """Retrieve prediction history"""
        if not self.is_available:
            return []
        try:
            entries = self.client.lrange(Config.REDIS_HISTORY_KEY, 0, limit - 1)
            return [PredictionHistoryEntry.model_validate_json(e) for e in entries]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
    
    def clear_history(self) -> bool:
        """Clear all history"""
        if not self.is_available:
            return False
        try:
            self.client.delete(Config.REDIS_HISTORY_KEY)
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False

redis_manager = RedisManager()

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
        self.calibration_data = None
        self.feature_importance = None
        self.dataset_stats = None
        self.dataset_cache = None  # Cached dataset for similarity search
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
                if Config.CALIBRATION_FILE.exists():
                    self.calibration_data = joblib.load(Config.CALIBRATION_FILE)
                if Config.FEATURE_IMPORTANCE_FILE.exists():
                    self.feature_importance = joblib.load(Config.FEATURE_IMPORTANCE_FILE)
                if Config.DATASET_STATS_FILE.exists():
                    self.dataset_stats = joblib.load(Config.DATASET_STATS_FILE)
                if Config.DATASET_CACHE_FILE.exists():
                    self.dataset_cache = joblib.load(Config.DATASET_CACHE_FILE)
                logger.info("✅ Model and artifacts loaded successfully from disk")
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
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            # 6. Calibration Curve
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
            calibration_data = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
                "sample_counts": [int(len(y_test) / 10)] * len(prob_true)
            }
            
            # 7. Feature Importance (Top 20)
            feature_importance = {
                "importances": clf.feature_importances_.tolist(),
                "top_20_indices": np.argsort(clf.feature_importances_)[-20:][::-1].tolist(),
                "top_20_scores": np.sort(clf.feature_importances_)[-20:][::-1].tolist()
            }
            
            # 8. Dataset Statistics
            ic50_values = df['value'].values
            mw_values = []
            logp_values = []
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    mw_values.append(Descriptors.MolWt(mol))
                    logp_values.append(Descriptors.MolLogP(mol))
            
            dataset_stats = {
                "total_samples": len(df),
                "active_count": int((y == 1).sum()),
                "inactive_count": int((y == 0).sum()),
                "class_balance": {
                    "active": float((y == 1).mean()),
                    "inactive": float((y == 0).mean())
                },
                "ic50_distribution": {
                    "min": float(np.min(ic50_values)),
                    "max": float(np.max(ic50_values)),
                    "median": float(np.median(ic50_values)),
                    "mean": float(np.mean(ic50_values)),
                    "q25": float(np.percentile(ic50_values, 25)),
                    "q75": float(np.percentile(ic50_values, 75))
                },
                "molecular_weight_stats": {
                    "min": float(np.min(mw_values)) if mw_values else 0,
                    "max": float(np.max(mw_values)) if mw_values else 0,
                    "mean": float(np.mean(mw_values)) if mw_values else 0,
                    "median": float(np.median(mw_values)) if mw_values else 0
                },
                "lipophilicity_stats": {
                    "min": float(np.min(logp_values)) if logp_values else 0,
                    "max": float(np.max(logp_values)) if logp_values else 0,
                    "mean": float(np.mean(logp_values)) if logp_values else 0,
                    "median": float(np.median(logp_values)) if logp_values else 0
                }
            }
            
            # 9. Cache dataset for similarity search
            dataset_cache = {
                "smiles": df['smiles'].tolist(),
                "activity": y.tolist(),
                "ic50": df['value'].tolist(),
                "chembl_ids": df['chembl_id'].tolist() if 'chembl_id' in df.columns else [None] * len(df)
            }
            
            logger.info(f"Training Results: {metrics}")
            
            # 10. Save all artifacts
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
                joblib.dump(calibration_data, Config.CALIBRATION_FILE)
                joblib.dump(feature_importance, Config.FEATURE_IMPORTANCE_FILE)
                joblib.dump(dataset_stats, Config.DATASET_STATS_FILE)
                joblib.dump(dataset_cache, Config.DATASET_CACHE_FILE)
                
                self.model = clf
                self.metadata = metadata
                self.calibration_data = calibration_data
                self.feature_importance = feature_importance
                self.dataset_stats = dataset_stats
                self.dataset_cache = dataset_cache
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Training finished in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
        finally:
            self.is_training = False

    @staticmethod
    def calculate_admet(mol, lipinski: LipinskiStats, extended: ExtendedDescriptors) -> ADMETDescriptors:
        """Calculate ADMET properties using rule-based predictions"""
        # Caco-2 Permeability (LogP-based proxy)
        caco = "High" if 0 < lipinski.LogP < 3 else "Low"
        
        # Human Intestinal Absorption (HIA) - Lipinski + TPSA
        hia = "Good" if extended.TPSA < 140 and lipinski.MW < 500 else "Poor"
        
        # Blood-Brain Barrier (BBB) Permeability
        # Rule: MW < 400, LogP 1-3, TPSA < 90, HBD < 3
        bbb = "Yes" if (lipinski.MW < 400 and 1 < lipinski.LogP < 3 and 
                       extended.TPSA < 90 and lipinski.NumHDonors < 3) else "No"
        
        # Plasma Protein Binding (high LogP = high binding)
        ppb = "High" if lipinski.LogP > 3 else "Low"
        
        # hERG Inhibition Risk (TPSA and LogP based)
        # Lower TPSA and higher LogP = higher risk
        herg = "High Risk" if (extended.TPSA < 100 and lipinski.LogP > 4) else "Low Risk"
        
        return ADMETDescriptors(
            CacoPermeability=caco,
            HumanIntestinalAbsorption=hia,
            BBBPermeant=bbb,
            PlasmaProteinBinding=ppb,
            hERG_Inhibition=herg
        )
    
    @staticmethod
    def calculate_drug_likeness(mol, lipinski: LipinskiStats, extended: ExtendedDescriptors) -> DrugLikenessScores:
        """Calculate comprehensive drug-likeness scores including QED, Lipinski, Veber, and PAINS"""
        
        # 1. QED Score (Quantitative Estimate of Drug-likeness)
        qed_score = round(qed(mol), 4)
        
        # QED Categories based on literature thresholds
        if qed_score >= 0.67:
            qed_category = "Excellent"
        elif qed_score >= 0.49:
            qed_category = "Good"
        elif qed_score >= 0.34:
            qed_category = "Moderate"
        else:
            qed_category = "Poor"
        
        # 2. Lipinski Rule of 5 Violations
        violations = 0
        if lipinski.MW > 500:
            violations += 1
        if lipinski.LogP > 5:
            violations += 1
        if lipinski.NumHDonors > 5:
            violations += 1
        if lipinski.NumHAcceptors > 10:
            violations += 1
        
        lipinski_compliant = violations <= 1  # Allow 1 violation
        
        # 3. Veber's Rules (for oral bioavailability)
        # TPSA <= 140 Ų AND Rotatable Bonds <= 10
        veber_compliant = extended.TPSA <= 140 and extended.NumRotatableBonds <= 10
        
        # 4. PAINS Filters (Pan-Assay Interference Compounds)
        pains_alerts = []
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(params)
            
            matches = catalog.GetMatches(mol)
            for match in matches:
                pains_alerts.append(match.GetDescription())
        except Exception as e:
            logger.warning(f"PAINS filter check failed: {e}")
        
        return DrugLikenessScores(
            qed=qed_score,
            qed_category=qed_category,
            lipinski_violations=violations,
            lipinski_compliant=lipinski_compliant,
            veber_compliant=veber_compliant,
            pains_alerts=pains_alerts,
            pains_count=len(pains_alerts)
        )
    
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
        
        # ADMET Predictions
        admet = self.calculate_admet(mol, lipinski, extended)
        
        # Drug-likeness Scores (QED, Lipinski, Veber, PAINS)
        drug_likeness = self.calculate_drug_likeness(mol, lipinski, extended)
        
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
        
        response = PredictionResponse(
            smiles=smiles,
            prediction=label,
            confidence=round(confidence, 4),
            probability=round(active_prob, 4),
            lipinski=lipinski,
            descriptors=extended,
            admet=admet,
            drug_likeness=drug_likeness,
            status="success",
            timestamp=datetime.now().isoformat()
        )
        
        # Add to history (non-blocking)
        try:
            history_entry = PredictionHistoryEntry(
                smiles=smiles,
                prediction=label,
                confidence=round(confidence, 4),
                probability=round(active_prob, 4),
                timestamp=response.timestamp
            )
            redis_manager.add_to_history(history_entry)
        except Exception as e:
            logger.warning(f"Failed to add to history: {e}")
        
        return response
    
    def similarity_search(self, query_smiles: str, top_k: int = 10, threshold: float = 0.7) -> List[SimilarCompound]:
        """Find similar compounds from training dataset"""
        if not self.dataset_cache:
            raise HTTPException(status_code=503, detail="Dataset cache not available")
        
        query_mol = Chem.MolFromSmiles(query_smiles)
        if not query_mol:
            raise HTTPException(status_code=400, detail="Invalid query SMILES")
        
        query_fp = self.mfpgen.GetFingerprint(query_mol)
        
        similarities = []
        for i, smiles in enumerate(self.dataset_cache['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = self.mfpgen.GetFingerprint(mol)
                sim = TanimotoSimilarity(query_fp, fp)
                
                if sim >= threshold and smiles != query_smiles:
                    similarities.append({
                        'smiles': smiles,
                        'similarity': sim,
                        'activity': self.dataset_cache['activity'][i],
                        'ic50_nm': self.dataset_cache['ic50'][i],
                        'chembl_id': self.dataset_cache['chembl_ids'][i]
                    })
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = []
        for sim in similarities[:top_k]:
            results.append(SimilarCompound(
                smiles=sim['smiles'],
                chembl_id=sim['chembl_id'],
                activity="Active" if sim['activity'] == 1 else "Inactive",
                similarity=round(sim['similarity'], 4),
                ic50_nm=round(sim['ic50_nm'], 2) if sim['ic50_nm'] else None
            ))
        
        return results

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

@app.get("/history", response_model=HistoryResponse)
async def get_prediction_history(limit: int = 50):
    """Get prediction history from Redis"""
    entries = redis_manager.get_history(limit=limit)
    return HistoryResponse(
        predictions=entries,
        count=len(entries),
        available=redis_manager.is_available
    )

@app.delete("/history")
async def clear_prediction_history():
    """Clear prediction history from Redis"""
    if not redis_manager.is_available:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    redis_manager.clear_history()
    return {"status": "success", "message": "History cleared"}

@app.post("/similarity/search", response_model=SimilaritySearchResponse)
async def search_similar_compounds(request: SimilaritySearchRequest):
    """Find similar compounds from training dataset"""
    results = model_manager.similarity_search(
        query_smiles=request.smiles,
        top_k=request.top_k or Config.SIMILARITY_TOP_K,
        threshold=request.threshold or Config.SIMILARITY_THRESHOLD
    )
    return SimilaritySearchResponse(
        query_smiles=request.smiles,
        results=results,
        count=len(results)
    )

@app.get("/model/calibration", response_model=CalibrationData)
async def get_calibration_data():
    """Get model calibration curve data"""
    if not model_manager.calibration_data:
        raise HTTPException(status_code=404, detail="Calibration data not available")
    return model_manager.calibration_data

@app.get("/model/feature-importance", response_model=FeatureImportance)
async def get_feature_importance():
    """Get top feature importances"""
    if not model_manager.feature_importance:
        raise HTTPException(status_code=404, detail="Feature importance not available")
    return model_manager.feature_importance

@app.get("/dataset/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """Get training dataset statistics"""
    if not model_manager.dataset_stats:
        raise HTTPException(status_code=404, detail="Dataset stats not available")
    return model_manager.dataset_stats

@app.post("/export/pdf")
async def export_prediction_pdf(prediction: PredictionResponse):
    """Generate PDF report for prediction"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    buffer = io.BytesIO()
    
    # Create PDF
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("OncoScope HER2 Prediction Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Prediction Summary
    pred_color = colors.HexColor('#10b981') if prediction.prediction == "Active" else colors.HexColor('#ef4444')
    pred_style = ParagraphStyle(
        'Prediction',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=pred_color,
        spaceAfter=20
    )
    story.append(Paragraph(f"Prediction: {prediction.prediction}", pred_style))
    story.append(Paragraph(f"Confidence: {prediction.confidence:.2%}", styles['Normal']))
    story.append(Paragraph(f"Probability (Active): {prediction.probability:.2%}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # SMILES
    story.append(Paragraph("<b>SMILES:</b>", styles['Heading3']))
    story.append(Paragraph(prediction.smiles, styles['Code']))
    story.append(Spacer(1, 0.2*inch))
    
    # Lipinski Properties
    story.append(Paragraph("<b>Lipinski Rule of 5:</b>", styles['Heading3']))
    lipinski_data = [
        ['Property', 'Value', 'Rule of 5 Limit'],
        ['Molecular Weight', f"{prediction.lipinski.MW:.2f} Da", '\u2264 500 Da'],
        ['LogP', f"{prediction.lipinski.LogP:.2f}", '\u2264 5'],
        ['H-Bond Donors', str(prediction.lipinski.NumHDonors), '\u2264 5'],
        ['H-Bond Acceptors', str(prediction.lipinski.NumHAcceptors), '\u2264 10']
    ]
    lipinski_table = Table(lipinski_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    lipinski_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(lipinski_table)
    story.append(Spacer(1, 0.2*inch))
    
    # ADMET Properties
    if prediction.admet:
        story.append(Paragraph("<b>ADMET Predictions:</b>", styles['Heading3']))
        admet_data = [
            ['Property', 'Prediction'],
            ['Caco-2 Permeability', prediction.admet.CacoPermeability],
            ['Human Intestinal Absorption', prediction.admet.HumanIntestinalAbsorption],
            ['BBB Permeant', prediction.admet.BBBPermeant],
            ['Plasma Protein Binding', prediction.admet.PlasmaProteinBinding],
            ['hERG Inhibition', prediction.admet.hERG_Inhibition]
        ]
        admet_table = Table(admet_data, colWidths=[3*inch, 3*inch])
        admet_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(admet_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Drug-Likeness Scores (QED, Lipinski, Veber, PAINS)
    if prediction.drug_likeness:
        story.append(Paragraph("<b>Drug-Likeness Assessment:</b>", styles['Heading3']))
        qed_color = colors.HexColor('#10b981') if prediction.drug_likeness.qed >= 0.67 else \
                   colors.HexColor('#3b82f6') if prediction.drug_likeness.qed >= 0.49 else \
                   colors.HexColor('#f59e0b') if prediction.drug_likeness.qed >= 0.34 else \
                   colors.HexColor('#ef4444')
        
        drug_likeness_data = [
            ['Metric', 'Value', 'Status'],
            ['QED Score', f"{prediction.drug_likeness.qed:.3f}", prediction.drug_likeness.qed_category],
            ['Lipinski Violations', str(prediction.drug_likeness.lipinski_violations), 
             'Compliant' if prediction.drug_likeness.lipinski_compliant else 'Non-compliant'],
            ["Veber's Rules", 'Pass' if prediction.drug_likeness.veber_compliant else 'Fail',
             'TPSA ≤ 140, RotB ≤ 10'],
            ['PAINS Alerts', str(prediction.drug_likeness.pains_count),
             'Clear' if prediction.drug_likeness.pains_count == 0 else 'Alerts Found']
        ]
        drug_likeness_table = Table(drug_likeness_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        drug_likeness_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#14b8a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(drug_likeness_table)
        
        # Add PAINS alerts if present
        if prediction.drug_likeness.pains_count > 0:
            story.append(Spacer(1, 0.1*inch))
            pains_style = ParagraphStyle(
                'PAINS',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#f59e0b')
            )
            story.append(Paragraph(
                f"<b>⚠️ PAINS Alerts:</b> {', '.join(prediction.drug_likeness.pains_alerts)}", 
                pains_style
            ))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Generated: {prediction.timestamp}", footer_style))
    story.append(Paragraph("OncoScope HER2 Drug Discovery Platform", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    
    # Return as base64
    pdf_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {
        "pdf": pdf_base64,
        "filename": f"prediction_{prediction.timestamp.replace(':', '-')}.pdf",
        "mime_type": "application/pdf"
    }

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

@app.post("/molecule/name-to-smiles", response_model=NameToSmilesResponse)
async def convert_name_to_smiles(payload: NameToSmilesRequest):
    """Convert a molecule name to SMILES using PubChem API"""
    name = payload.name.strip()
    
    # First check if it's already a valid SMILES
    mol = Chem.MolFromSmiles(name)
    if mol:
        return NameToSmilesResponse(
            name=name,
            smiles=name,
            status="success"
        )
    
    try:
        # Query PubChem API
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try compound name first - request both IsomericSMILES and CanonicalSMILES
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES,CanonicalSMILES/JSON"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                # Try different SMILES fields
                smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
                
                if smiles:
                    # Validate the SMILES with RDKit
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        return NameToSmilesResponse(
                            name=name,
                            smiles=smiles,
                            status="success"
                        )
            
            # Try getting CID first, then SMILES
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID", [])
                if cids:
                    cid = cids[0]
                    # Get SMILES from CID
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                        smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
                        
                        if smiles:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                return NameToSmilesResponse(
                                    name=name,
                                    smiles=smiles,
                                    status="success"
                                )
        
        raise HTTPException(status_code=404, detail=f"Could not find molecule: {name}")
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="PubChem API timeout. Please try again.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Name to SMILES conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert name to SMILES: {str(e)}")

@app.post("/export/svg", response_model=MoleculeImageExportResponse)
async def export_molecule_svg(payload: MoleculeImageExportRequest):
    """Export molecule as SVG image file"""
    mol = Chem.MolFromSmiles(payload.smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create SVG drawer
        drawer = Draw.MolDraw2DSVG(payload.width, payload.height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False
        
        # Set background
        if payload.background == "transparent":
            drawer.drawOptions().clearBackground = False
        elif payload.background != "white":
            # Custom color (hex)
            try:
                drawer.drawOptions().setBackgroundColour((1, 1, 1, 1))  # Default white
            except:
                pass
        
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_content = drawer.GetDrawingText()
        
        # Encode as base64
        content_b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return MoleculeImageExportResponse(
            smiles=payload.smiles,
            format="svg",
            content=content_b64,
            filename=f"molecule_{timestamp}.svg",
            mime_type="image/svg+xml",
            width=payload.width,
            height=payload.height
        )
        
    except Exception as e:
        logger.error(f"SVG export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export SVG: {str(e)}")

@app.post("/export/png", response_model=MoleculeImageExportResponse)
async def export_molecule_png(payload: MoleculeImageExportRequest):
    """Export molecule as PNG image file"""
    mol = Chem.MolFromSmiles(payload.smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    try:
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create PNG using MolToImage
        from PIL import Image
        img = Draw.MolToImage(mol, size=(payload.width, payload.height))
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode as base64
        content_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return MoleculeImageExportResponse(
            smiles=payload.smiles,
            format="png",
            content=content_b64,
            filename=f"molecule_{timestamp}.png",
            mime_type="image/png",
            width=payload.width,
            height=payload.height
        )
        
    except Exception as e:
        logger.error(f"PNG export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export PNG: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)