# OncoScope — AI-Powered HER2 Drug Discovery Platform

<p align="center">
  <strong>Enterprise-Grade Machine Learning Platform for Bioactivity Prediction and Molecular Analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js 16">
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react" alt="React 19">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/RDKit-2023.9+-2ECC71?style=flat-square" alt="RDKit">
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
  - [Purpose and Core Objective](#purpose-and-core-objective)
  - [Problem Statement](#problem-statement)
  - [End-to-End System Workflow](#end-to-end-system-workflow)
- [Key Features](#key-features)
  - [Frontend Features (Next.js)](#frontend-features-nextjs)
  - [Backend Features (FastAPI)](#backend-features-fastapi)
  - [Machine Learning Features](#machine-learning-features)
- [Technology Stack](#technology-stack)
- [Setup & Execution Guide](#setup--execution-guide)
  - [Prerequisites](#prerequisites)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
- [Application Architecture](#application-architecture)
  - [Frontend Architecture](#frontend-architecture)
  - [Backend Architecture](#backend-architecture)
  - [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Data Flow & Processing](#data-flow--processing)
- [API Reference](#api-reference)
- [Potential Enhancements (Future Scope)](#potential-enhancements-future-scope)
- [Production Readiness Notes](#production-readiness-notes)

---

## Project Overview

### Purpose and Core Objective

OncoScope is a full-stack drug discovery application designed to predict the bioactivity of chemical compounds against the **HER2 (Human Epidermal Growth Factor Receptor 2)** target—a critical oncogene implicated in aggressive forms of breast cancer. The platform leverages machine learning to accelerate early-stage drug discovery by enabling researchers to rapidly screen potential HER2 inhibitors.

The system provides:
- **Binary Classification**: Predicts whether a molecule is "Active" or "Inactive" against HER2
- **Probability Scoring**: Returns confidence levels and probability scores for predictions
- **Drug-Likeness Assessment**: Evaluates compounds using Lipinski's Rule of 5, Veber's rules, QED scores, and PAINS filters
- **ADMET Predictions**: Provides rule-based pharmacokinetic property estimates
- **Molecular Visualization**: Renders 2D molecular structures from SMILES notation
- **Report Generation**: Exports comprehensive PDF reports for research documentation

### Problem Statement

Drug discovery is an expensive, time-consuming process where early-stage screening of compound libraries against biological targets can take months. Traditional high-throughput screening (HTS) is resource-intensive, requiring physical synthesis and testing of thousands of compounds.

OncoScope addresses this challenge by providing:
1. **Virtual Screening Capability**: Predict bioactivity computationally before wet-lab synthesis
2. **Instant Feedback**: Sub-second prediction times for molecular analysis
3. **Comprehensive Analysis**: Beyond simple active/inactive classification, provides drug-likeness metrics essential for lead optimization
4. **Accessibility**: User-friendly interface that doesn't require computational chemistry expertise

### End-to-End System Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Input: SMILES string (e.g., "CCO") OR Molecule name (e.g., "Erlotinib") │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js)                                  │
│  1. User types compound name or SMILES in search box                            │
│  2. Smart detection: Determines if input is SMILES or molecule name             │
│  3. If molecule name → API call to /molecule/name-to-smiles                     │
│  4. Validated SMILES → API call to /predict                                     │
│  5. Display loading state while awaiting response                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (FastAPI)                                   │
│  1. Receive SMILES string via POST /predict                                     │
│  2. Validate SMILES using RDKit (Chem.MolFromSmiles)                           │
│  3. Calculate molecular descriptors (Lipinski, extended, ADMET)                 │
│  4. Generate Morgan fingerprint (radius=2, 2048 bits)                           │
│  5. Load RandomForest model and run inference                                   │
│  6. Calculate drug-likeness scores (QED, PAINS, Veber)                         │
│  7. Return structured JSON response                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ML INFERENCE                                        │
│  Model: RandomForestClassifier (scikit-learn)                                   │
│  Input: 2048-bit Morgan fingerprint vector                                      │
│  Output: [P(Inactive), P(Active)] probability array                             │
│  Decision: Active if P(Active) >= 0.5                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Visualization)                            │
│  1. Parse PredictionResponse JSON                                               │
│  2. Render prediction card (Active/Inactive with confidence)                    │
│  3. Fetch and display 2D molecule structure (SVG from backend)                  │
│  4. Visualize Lipinski properties via radar chart                               │
│  5. Display extended descriptors via bar chart                                  │
│  6. Show QED score, compliance badges, PAINS alerts                             │
│  7. Enable PDF export of complete analysis                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Frontend Features (Next.js)

#### User Interface & Interactions

| Feature | Description |
|---------|-------------|
| **Smart Search Input** | Combined search box accepting both SMILES strings and molecule names. Intelligently detects input type using pattern matching (checks for SMILES-specific characters like `[]()=#@/\\`). |
| **Compound Database Dropdown** | Pre-populated list of 30 known compounds (13 active HER2 inhibitors, 17 inactive compounds) from CHEMBL1824 dataset. Includes compound name, SMILES preview, activity status, and IC50 values. |
| **PubChem Integration** | Automatic molecule name-to-SMILES conversion via backend PubChem API lookup when compound not in local database. |
| **Real-time System Status** | Live backend health monitoring with visual status badge (Online/Offline). Health check runs every 30 seconds. |
| **Dark/Light Theme Toggle** | Full dark mode support with system preference detection and manual override. Theme persists across sessions. |

#### Data Visualization Components

| Component | Library | Purpose |
|-----------|---------|---------|
| **Lipinski Radar Chart** | Recharts `RadarChart` | Visualizes Lipinski Rule of 5 compliance. Normalizes MW, LogP, HBD, HBA to percentage of threshold values. Color-coded to show violations. |
| **Descriptors Bar Chart** | Recharts `BarChart` | Displays extended molecular properties (TPSA, rotatable bonds, aromatic rings, heavy atoms, Fsp³). Color indicates optimal vs. warning ranges. |
| **Prediction Result Card** | Custom component | Large visual display of Active/Inactive prediction with confidence percentage and probability bar. Color-coded (teal for active, rose for inactive). |
| **QED Score Display** | Custom component | Quantitative Estimate of Drug-likeness (0-1) with category labels (Excellent/Good/Moderate/Poor) and progress bar. |
| **Compliance Badges** | Custom grid | Three-column display showing Lipinski, Veber, and PAINS compliance status with Pass/Fail indicators. |

#### Molecular Visualization

| Feature | Implementation |
|---------|---------------|
| **2D Structure Viewer** | Fetches SVG from backend `/molecule/svg` endpoint (RDKit server-side rendering). Supports click-to-enlarge modal view. |
| **Molecular Formula Display** | Shows calculated molecular formula (e.g., C₂₅H₂₄ClFN₄O₃) from backend descriptors. |
| **SMILES Display** | Monospace formatted SMILES string with truncation for long molecules. |

#### Error Handling & Loading States

| State | Implementation |
|-------|---------------|
| **Loading State** | Skeleton-based loading animation using custom `LoadingState` component. Displays centered spinner with "Analyzing..." text. |
| **Empty State** | Placeholder component with icon, title, and description when no prediction has been run. |
| **Error Toasts** | Sonner toast notifications for API errors, validation failures, and network issues. Includes specific error messages from backend. |
| **Offline Detection** | Status badge changes to "Offline" when backend health check fails. Prediction button disabled. |

#### Export Capabilities

| Feature | Format | Details |
|---------|--------|---------|
| **PDF Report Export** | PDF | Complete prediction report including SMILES, prediction result, Lipinski properties, ADMET predictions, drug-likeness scores, and PAINS alerts. Generated server-side using ReportLab. |
| **Molecule Image Export** | SVG/PNG | Export 2D molecular structure as image file (accessible via API). |

### Backend Features (FastAPI)

#### API Endpoints

| Endpoint | Method | Purpose | Request Body | Response |
|----------|--------|---------|--------------|----------|
| `/health` | GET | System health and model status | — | `HealthResponse` with model_ready boolean, version, metrics |
| `/predict` | POST | Main bioactivity prediction | `{ smiles: string }` | `PredictionResponse` with prediction, confidence, descriptors |
| `/train` | POST | Trigger background model retraining | — | `TrainingResponse` with status |
| `/model/info` | GET | Model configuration and metrics | — | `ModelInfoResponse` with metadata |
| `/molecule/svg` | POST | Generate 2D structure SVG | `{ smiles, width, height }` | `MoleculeSVGResponse` with SVG string |
| `/molecule/name-to-smiles` | POST | Convert drug name to SMILES | `{ name: string }` | `NameToSmilesResponse` |
| `/similarity/search` | POST | Find similar compounds | `{ smiles, top_k, threshold }` | `SimilaritySearchResponse` |
| `/history` | GET | Retrieve prediction history | Query param: `limit` | `HistoryResponse` (Redis-backed) |
| `/history` | DELETE | Clear prediction history | — | Success message |
| `/model/calibration` | GET | Calibration curve data | — | `CalibrationData` |
| `/model/feature-importance` | GET | Feature importance scores | — | `FeatureImportance` |
| `/dataset/stats` | GET | Training dataset statistics | — | `DatasetStats` |
| `/export/pdf` | POST | Generate PDF report | `PredictionResponse` | Base64-encoded PDF |
| `/export/svg` | POST | Export molecule as SVG | `{ smiles, width, height }` | Base64-encoded SVG |
| `/export/png` | POST | Export molecule as PNG | `{ smiles, width, height }` | Base64-encoded PNG |

#### Request Validation

- **Pydantic v2 Models**: All request/response bodies validated using Pydantic `BaseModel` classes
- **SMILES Validation**: RDKit `Chem.MolFromSmiles()` check before processing; returns HTTP 400 for invalid SMILES
- **Field Constraints**: `Field()` validators for min/max lengths, value ranges (e.g., image dimensions 100-800px)

#### Error Handling

| Error Type | HTTP Code | Handling |
|------------|-----------|----------|
| Invalid SMILES | 400 | `HTTPException` with descriptive message |
| Model not ready | 503 | Service unavailable response |
| PubChem timeout | 504 | Gateway timeout with retry suggestion |
| Internal errors | 500 | Global exception handler with logging |
| Validation errors | 422 | Pydantic validation errors with field details |

#### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurable for production
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Machine Learning Features

#### Model Specification

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Classifier (scikit-learn) |
| **Estimators** | 200 trees |
| **Max Depth** | None (nodes expanded until leaves are pure) |
| **Class Weighting** | Balanced (automatic adjustment for class imbalance) |
| **Input Features** | 2048-bit Morgan fingerprint (ECFP4-like) |
| **Output** | Binary classification (0=Inactive, 1=Active) + probability scores |

#### Feature Engineering

| Feature Type | Implementation |
|--------------|---------------|
| **Morgan Fingerprints** | RDKit `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` |
| **Fingerprint Radius** | 2 (captures up to 4-bond neighborhoods) |
| **Fingerprint Size** | 2048 bits (fixed-length binary vector) |

#### Training Data

| Property | Value |
|----------|-------|
| **Source** | ChEMBL database (CHEMBL1824 - HER2/ErbB2) |
| **File** | `backend/data/CHEMBL1824_bioactivity.csv` (semicolon-separated) |
| **Activity Threshold** | IC50 ≤ 1000 nM = Active; IC50 > 1000 nM = Inactive |
| **Preprocessing** | Unit normalization (µM→nM), ambiguous data removal, duplicate handling |

#### Model Output Format

```json
{
  "smiles": "CCO",
  "prediction": "Active",
  "confidence": 0.8742,
  "probability": 0.8742,
  "lipinski": { "MW": 46.07, "LogP": -0.31, "NumHDonors": 1, "NumHAcceptors": 1 },
  "descriptors": { "TPSA": 20.23, "NumRotatableBonds": 0, ... },
  "admet": { "CacoPermeability": "High", "BBBPermeant": "Yes", ... },
  "drug_likeness": { "qed": 0.407, "lipinski_violations": 0, "pains_alerts": [], ... },
  "status": "success",
  "timestamp": "2024-12-19T10:30:00.000Z"
}
```

#### Drug-Likeness Scoring

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **QED Score** | Quantitative Estimate of Drug-likeness (0-1) | RDKit `Chem.QED.qed(mol)` |
| **Lipinski Violations** | Count of Rule of 5 violations (0-4) | Manual threshold checks |
| **Veber's Rules** | Oral bioavailability predictor | TPSA ≤ 140 AND RotBonds ≤ 10 |
| **PAINS Filters** | Pan-Assay Interference detection | RDKit `FilterCatalog` with PAINS catalog |

#### ADMET Predictions

| Property | Prediction Logic |
|----------|-----------------|
| **Caco-2 Permeability** | High if 0 < LogP < 3 |
| **Human Intestinal Absorption** | Good if TPSA < 140 AND MW < 500 |
| **BBB Permeability** | Yes if MW < 400 AND 1 < LogP < 3 AND TPSA < 90 AND HBD < 3 |
| **Plasma Protein Binding** | High if LogP > 3 |
| **hERG Inhibition Risk** | High Risk if TPSA < 100 AND LogP > 4 |

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Next.js** | 16.0.10 | React framework with App Router |
| **React** | 19.2.1 | UI library |
| **TypeScript** | 5.x | Type-safe JavaScript |
| **Tailwind CSS** | 4.x | Utility-first CSS framework |
| **shadcn/ui** | 3.6.1 | Accessible component primitives |
| **Recharts** | 3.6.0 | Charting library for data visualization |
| **Lucide React** | 0.561.0 | Icon library |
| **Sonner** | 2.0.7 | Toast notifications |
| **next-themes** | 0.4.6 | Dark/light theme management |
| **Framer Motion** | 12.23.26 | Animation library |
| **class-variance-authority** | 0.7.1 | Component variant utility |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | 0.104+ | Async Python web framework |
| **Uvicorn** | 0.24+ | ASGI server |
| **Pydantic** | 2.x | Data validation and serialization |
| **Python** | 3.10+ | Runtime |
| **httpx** | 0.25+ | Async HTTP client (PubChem API) |
| **Redis** | 5.0+ | Prediction history caching (optional) |
| **ReportLab** | 4.0+ | PDF generation |
| **Pillow** | 10.0+ | Image processing for PNG export |

### Machine Learning & Cheminformatics

| Technology | Version | Purpose |
|------------|---------|---------|
| **scikit-learn** | 1.3+ | RandomForest classifier, metrics |
| **RDKit** | 2023.9.1+ | Cheminformatics (fingerprints, descriptors, 2D rendering) |
| **NumPy** | 1.24+ | Numerical operations |
| **Pandas** | 2.0+ | Data processing |
| **joblib** | 1.3+ | Model serialization |

### Infrastructure & Tooling

| Tool | Purpose |
|------|---------|
| **npm** | Frontend package management |
| **pip** | Python package management |
| **ESLint** | JavaScript/TypeScript linting |
| **PostCSS** | CSS processing pipeline |

---

## Setup & Execution Guide

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Node.js** | 18.x or 20.x+ | LTS version recommended |
| **npm** | 9.x+ | Comes with Node.js |
| **Python** | 3.10+ | Required for RDKit compatibility |
| **pip** | Latest | Python package installer |
| **Redis** | 5.0+ | Optional (for prediction history) |

### Running the Backend

#### 1. Create and activate virtual environment

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Start the FastAPI server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# OR using the main.py entry point
python main.py
```

#### 4. Model initialization

On first startup, if `models/her2_rf_model.joblib` doesn't exist:
- Training pipeline automatically triggers in background thread
- Loads `data/CHEMBL1824_bioactivity.csv`
- Trains RandomForest model and saves artifacts
- Training typically completes in 10-30 seconds

#### 5. Verify backend is running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "online",
  "model_ready": true,
  "version": "4.0.0",
  "backend": "RandomForest (scikit-learn)"
}
```

#### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_PASSWORD` | `None` | Redis authentication password |

### Running the Frontend

#### 1. Install dependencies

```bash
cd frontend
npm install
```

#### 2. Configure environment (optional)

Create `.env.local` if backend runs on non-default port:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

#### 3. Start development server

```bash
npm run dev
```

Frontend available at: `http://localhost:3000`

#### 4. Production build

```bash
# Build optimized production bundle
npm run build

# Start production server
npm start
```

---

## Application Architecture

### Frontend Architecture

```
frontend/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # Root layout with ThemeProvider, Toaster
│   ├── page.tsx                 # Main dashboard (single-page app)
│   └── globals.css              # Global styles, CSS variables
├── components/
│   ├── ui/                      # shadcn/ui primitives + custom components
│   │   ├── button.tsx           # Button variants
│   │   ├── card.tsx             # Card container
│   │   ├── input.tsx            # Form input
│   │   ├── progress.tsx         # Progress bar
│   │   ├── skeleton.tsx         # Loading skeleton
│   │   ├── molecule-viewer.tsx  # 2D structure renderer (backend SVG)
│   │   ├── status-badge.tsx     # Online/offline indicator
│   │   ├── loading-state.tsx    # Loading skeleton patterns
│   │   ├── empty-state.tsx      # Placeholder component
│   │   └── export-pdf-button.tsx# PDF export trigger
│   ├── charts/
│   │   ├── lipinski-radar-chart.tsx   # Recharts radar chart
│   │   └── descriptors-bar-chart.tsx  # Recharts bar chart
│   └── theme-provider.tsx       # next-themes provider wrapper
├── lib/
│   ├── api.ts                   # API client (fetch wrappers)
│   └── utils.ts                 # Utility functions (cn, etc.)
└── types/
    └── api.ts                   # TypeScript interfaces mirroring backend
```

#### Design Patterns

| Pattern | Implementation |
|---------|---------------|
| **Single Page Application** | All functionality on `page.tsx`; no routing required |
| **Component Composition** | Small, focused components combined in bento grid layout |
| **Type Safety** | Full TypeScript coverage with shared API types |
| **API Abstraction** | Centralized `api.ts` client; components never use raw fetch |
| **Theme System** | CSS variables + next-themes for consistent dark mode |

### Backend Architecture

```
backend/
├── main.py                      # Single-file FastAPI application
│   ├── Config                   # Centralized configuration class
│   ├── Pydantic Models          # Request/response schemas
│   ├── RedisManager             # Redis connection and history operations
│   ├── DataProcessor            # Data loading and cleaning pipeline
│   ├── ModelManager             # ML model lifecycle (load/train/predict)
│   └── API Endpoints            # FastAPI route handlers
├── data/
│   └── CHEMBL1824_bioactivity.csv  # Training dataset
├── models/
│   ├── her2_rf_model.joblib     # Trained RandomForest model
│   ├── model_metadata.joblib    # Training metrics and config
│   ├── calibration_data.joblib  # Calibration curve data
│   ├── feature_importance.joblib # Top feature importances
│   ├── dataset_stats.joblib     # Dataset statistics
│   └── dataset_cache.joblib     # Cached data for similarity search
└── requirements.txt             # Python dependencies
```

#### Design Patterns

| Pattern | Implementation |
|---------|---------------|
| **Single-File Architecture** | All logic in `main.py` for simplicity |
| **Thread-Safe Model Access** | `threading.RLock()` for concurrent prediction safety |
| **Background Training** | Training runs in separate thread; doesn't block API |
| **Lifespan Management** | FastAPI lifespan context for startup/shutdown |
| **Global Singletons** | `model_manager` and `redis_manager` instances |

### ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                     │
│  Input: CHEMBL1824_bioactivity.csv (semicolon-separated)                    │
│  Columns: Molecule ChEMBL ID, Smiles, Standard Value, Standard Units        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA CLEANING                                      │
│  1. Filter by allowed units (nM, uM)                                        │
│  2. Convert uM → nM (multiply by 1000)                                      │
│  3. Handle relations ('<', '>', '=', '~') for activity labeling             │
│  4. Remove ambiguous cases (mixed signals for same SMILES)                  │
│  5. Deduplicate by SMILES                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING                                  │
│  For each valid SMILES:                                                     │
│  1. Parse with RDKit: mol = Chem.MolFromSmiles(smiles)                      │
│  2. Generate fingerprint: mfpgen.GetFingerprint(mol)                        │
│  3. Convert to numpy array: 2048-dimensional binary vector                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL TRAINING                                     │
│  1. Train/test split: 80/20, stratified by class                            │
│  2. RandomForestClassifier(n_estimators=200, class_weight='balanced')       │
│  3. Fit on training data                                                    │
│  4. Evaluate: accuracy, precision, recall, F1                               │
│  5. Generate calibration curve (10 bins)                                    │
│  6. Extract top 20 feature importances                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL PERSISTENCE                                   │
│  joblib.dump() for all artifacts:                                           │
│  - her2_rf_model.joblib (trained classifier)                                │
│  - model_metadata.joblib (metrics, config, timestamp)                       │
│  - calibration_data.joblib (prob_true, prob_pred arrays)                    │
│  - feature_importance.joblib (indices, scores)                              │
│  - dataset_stats.joblib (class balance, IC50 distribution)                  │
│  - dataset_cache.joblib (SMILES, activities for similarity search)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow & Processing

### Prediction Request Flow

```
1. USER INPUT
   ├── Option A: SMILES string (e.g., "CCO")
   │   └── Frontend sends directly to /predict
   └── Option B: Molecule name (e.g., "Erlotinib")
       └── Frontend calls /molecule/name-to-smiles first
           └── Backend queries PubChem REST API
               └── Returns canonical SMILES → then calls /predict

2. BACKEND PROCESSING (/predict endpoint)
   ├── SMILES Validation
   │   └── mol = Chem.MolFromSmiles(smiles)
   │   └── If None → HTTP 400 "Invalid SMILES"
   │
   ├── Descriptor Calculation (parallel)
   │   ├── Lipinski: MW, LogP, HBD, HBA
   │   ├── Extended: TPSA, RotBonds, AromaticRings, Fsp3, HeavyAtoms, Formula
   │   ├── ADMET: Rule-based predictions
   │   └── Drug-likeness: QED, Lipinski violations, Veber, PAINS
   │
   ├── Fingerprint Generation
   │   └── Morgan fingerprint → 2048-bit numpy array
   │
   └── Model Inference
       ├── model.predict_proba(fingerprint)
       ├── Extract P(Active) from probability array
       ├── Threshold at 0.5 for binary classification
       └── Confidence = max(P(Active), P(Inactive))

3. RESPONSE CONSTRUCTION
   └── PredictionResponse {
         smiles, prediction, confidence, probability,
         lipinski, descriptors, admet, drug_likeness,
         status, timestamp
       }

4. FRONTEND RENDERING
   ├── Update result state
   ├── Render prediction card with color-coded result
   ├── Trigger molecule SVG fetch (/molecule/svg)
   ├── Render Lipinski radar chart
   ├── Render descriptors bar chart
   ├── Display QED score and compliance badges
   └── Show PAINS alerts if present
```

### Molecule Name Resolution Flow

```
/molecule/name-to-smiles

1. Check if input is already valid SMILES
   └── If yes → return immediately

2. Query PubChem (primary method)
   └── GET pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON

3. If not found, try CID lookup
   └── GET /compound/name/{name}/cids/JSON → Get CID
   └── GET /compound/cid/{cid}/property/IsomericSMILES/JSON → Get SMILES

4. Validate returned SMILES with RDKit

5. Return NameToSmilesResponse or HTTP 404
```

---

## API Reference

### Core Endpoints

#### `POST /predict`
Predict HER2 bioactivity for a molecule.

**Request:**
```json
{ "smiles": "CCO" }
```

**Response (200):**
```json
{
  "smiles": "CCO",
  "prediction": "Inactive",
  "confidence": 0.9234,
  "probability": 0.0766,
  "lipinski": {
    "MW": 46.07,
    "LogP": -0.31,
    "NumHDonors": 1,
    "NumHAcceptors": 1
  },
  "descriptors": {
    "MW": 46.07,
    "LogP": -0.31,
    "NumHDonors": 1,
    "NumHAcceptors": 1,
    "TPSA": 20.23,
    "NumRotatableBonds": 0,
    "NumAromaticRings": 0,
    "FractionCSP3": 0.5,
    "NumHeavyAtoms": 3,
    "MolecularFormula": "C2H6O"
  },
  "admet": {
    "CacoPermeability": "High",
    "HumanIntestinalAbsorption": "Good",
    "BBBPermeant": "Yes",
    "PlasmaProteinBinding": "Low",
    "hERG_Inhibition": "Low Risk"
  },
  "drug_likeness": {
    "qed": 0.407,
    "qed_category": "Moderate",
    "lipinski_violations": 0,
    "lipinski_compliant": true,
    "veber_compliant": true,
    "pains_alerts": [],
    "pains_count": 0
  },
  "status": "success",
  "timestamp": "2024-12-19T10:30:00.000000"
}
```

#### `GET /health`
Check system health and model status.

**Response (200):**
```json
{
  "status": "online",
  "model_ready": true,
  "version": "4.0.0",
  "backend": "RandomForest (scikit-learn)",
  "model_info": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "f1": 0.89,
    "training_samples": 1200,
    "test_samples": 300
  },
  "timestamp": "2024-12-19T10:30:00.000000"
}
```

#### `POST /molecule/name-to-smiles`
Convert molecule name to SMILES notation.

**Request:**
```json
{ "name": "Erlotinib" }
```

**Response (200):**
```json
{
  "name": "Erlotinib",
  "smiles": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
  "status": "success"
}
```

---

## Potential Enhancements (Future Scope)

> **Note:** All items below are **future improvements**, not current features.

### Frontend Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Batch Prediction UI** | Allow users to upload CSV files with multiple SMILES for batch processing |
| **3D Molecule Viewer** | Integrate 3Dmol.js or NGL for interactive 3D molecular visualization |
| **Prediction History Panel** | Sidebar showing recent predictions with quick-access recall |
| **Comparison Mode** | Side-by-side comparison of two molecules with difference highlighting |
| **Accessibility Improvements** | WCAG 2.1 AA compliance, screen reader optimization, keyboard navigation |
| **Mobile Responsive Design** | Optimize bento grid layout for tablet and mobile viewports |
| **Real-time Collaboration** | Share prediction sessions via unique URLs |
| **Internationalization (i18n)** | Multi-language support for global research teams |

### Backend Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Authentication & Authorization** | JWT-based auth with role-based access control (RBAC) |
| **Rate Limiting** | Redis-backed rate limiting to prevent API abuse |
| **API Versioning** | URL-based versioning (e.g., `/api/v1/predict`) |
| **Batch Prediction Endpoint** | Async batch processing with job queue (Celery/Redis) |
| **WebSocket Support** | Real-time training progress updates |
| **Model A/B Testing** | Serve multiple model versions with traffic splitting |
| **Caching Layer** | Redis caching for repeated predictions (fingerprint hash key) |
| **OpenAPI Enhancement** | Detailed endpoint documentation with examples |
| **Database Integration** | PostgreSQL for persistent prediction storage |

### Machine Learning Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Deep Learning Models** | Graph Neural Networks (GNN) or transformer-based architectures |
| **Ensemble Methods** | Combine RF with gradient boosting, neural networks |
| **Automated Retraining** | Scheduled retraining with new ChEMBL data releases |
| **Model Explainability (XAI)** | SHAP values, attention maps for prediction interpretation |
| **Uncertainty Quantification** | Conformal prediction for calibrated confidence intervals |
| **Multi-Task Learning** | Predict activity against multiple kinase targets simultaneously |
| **Active Learning** | Suggest most informative compounds for experimental validation |
| **Transfer Learning** | Pre-train on large molecular datasets, fine-tune on HER2 |

### Infrastructure Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Containerization** | Docker images for frontend and backend |
| **Kubernetes Deployment** | Helm charts for scalable cloud deployment |
| **CI/CD Pipeline** | GitHub Actions for automated testing and deployment |
| **Monitoring & Observability** | Prometheus metrics, Grafana dashboards, structured logging |
| **CDN Integration** | Edge caching for static assets |

---

## Production Readiness Notes

### Current Limitations

| Area | Limitation | Impact |
|------|------------|--------|
| **Authentication** | No user authentication implemented | Open access; not suitable for sensitive data |
| **CORS Policy** | Allows all origins (`*`) | Should be restricted to specific domains in production |
| **Rate Limiting** | Not implemented | Vulnerable to API abuse and DoS |
| **HTTPS** | Not enforced by application | Must be handled by reverse proxy (nginx, Traefik) |
| **Secrets Management** | No dedicated secrets vault | Redis password in environment variable |
| **Model Versioning** | Single model version active | No rollback capability |

### Scalability Considerations

| Component | Current State | Production Recommendation |
|-----------|--------------|---------------------------|
| **Backend** | Single Uvicorn process | Multiple workers with Gunicorn; horizontal scaling behind load balancer |
| **Model Loading** | Loaded at startup per process | Consider model serving platform (TensorFlow Serving, Triton) for multi-worker deployments |
| **Redis** | Optional, single instance | Redis Sentinel or Cluster for high availability |
| **Fingerprint Calculation** | Synchronous | Async task queue for batch predictions |
| **Frontend** | Static export possible | CDN deployment with edge caching |

### Security Considerations

| Risk | Mitigation Required |
|------|---------------------|
| **SMILES Injection** | RDKit parsing provides implicit sanitization; add additional input length limits |
| **PubChem API Dependency** | Implement circuit breaker pattern; cache known molecule names |
| **Denial of Service** | Implement rate limiting (10-100 requests/minute per IP) |
| **Data Privacy** | If storing predictions, implement data retention policies and encryption at rest |
| **Dependency Vulnerabilities** | Regular `pip audit` and `npm audit` in CI pipeline |

### Monitoring & Logging Gaps

| Missing Capability | Recommendation |
|--------------------|----------------|
| **Structured Logging** | Replace print/logger.info with JSON logging (structlog) |
| **Request Tracing** | Add correlation IDs for request tracking |
| **Metrics Collection** | Prometheus metrics for latency, prediction distribution, error rates |
| **Alerting** | Set up alerts for high error rates, model staleness |
| **Health Checks** | Kubernetes-compatible liveness/readiness probes |

### Deployment Readiness Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Functional Completeness** | ✅ Ready | Core prediction workflow complete |
| **Error Handling** | ✅ Ready | Comprehensive exception handling |
| **Documentation** | ✅ Ready | API documented, README complete |
| **Testing** | ⚠️ Partial | Unit tests not included; recommend pytest coverage |
| **Security** | ⚠️ Partial | Authentication and rate limiting required |
| **Observability** | ⚠️ Partial | Basic logging only; monitoring needed |
| **Scalability** | ⚠️ Partial | Works single-instance; multi-worker requires changes |
| **CI/CD** | ❌ Not Ready | Pipeline not configured |
| **Containerization** | ❌ Not Ready | Dockerfiles not present |

---

<p align="center">
  <strong>OncoScope</strong> — Accelerating Drug Discovery Through AI<br>
  <em>Research Use Only • Not for Clinical Decision Making</em>
</p>
