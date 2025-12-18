# Copilot Instructions – OncoScope HER2 Drug Discovery Platform

## Architecture Overview

This is a **full-stack drug discovery application** for predicting HER2 bioactivity of molecules:

- **Backend** (`backend/`): FastAPI + RDKit + scikit-learn RandomForest classifier
- **Frontend** (`frontend/`): Next.js 16 + React 19 + Tailwind CSS 4 + shadcn/ui components

**Data Flow**: User inputs SMILES string or molecule name → Backend converts name→SMILES via PubChem API → RDKit generates Morgan fingerprints → RandomForest predicts Active/Inactive → Returns Lipinski descriptors + confidence scores.

## Running the Project

```bash
# Backend (port 8000)
cd backend && python main.py  # or: uvicorn main:app --reload

# Frontend (port 3000)
cd frontend && npm run dev
```

Backend auto-trains the model on first start if `models/her2_rf_model.joblib` doesn't exist.

## Key Patterns & Conventions

### Backend (`backend/main.py`)
- **Single-file architecture**: All logic in `main.py` – Config class, Pydantic models, DataProcessor, ModelManager, and FastAPI routes
- **Pydantic models match frontend types**: Keep `LipinskiStats`, `ExtendedDescriptors`, `PredictionResponse` in sync with `frontend/types/api.ts`
- **Thread-safe model access**: Use `ModelManager.lock` for model operations; training runs in background threads
- **SMILES validation**: Always use `Chem.MolFromSmiles()` and check for `None` before processing

### Frontend
- **API client pattern**: All backend calls go through `lib/api.ts` – never use raw `fetch` in components
- **Type definitions**: All API response types live in `types/api.ts` – mirror backend Pydantic models exactly
- **UI components**: Use shadcn/ui primitives from `components/ui/` – see `button.tsx`, `card.tsx` as examples
- **Charts**: Recharts-based visualizations in `components/charts/` with dark mode support via CSS variables
- **Molecule rendering**: `MoleculeViewer` component calls backend `/molecule/svg` endpoint (server-side RDKit, not WASM)

### Type Synchronization (Critical)
When modifying API response shapes, update both:
1. `backend/main.py` – Pydantic model classes
2. `frontend/types/api.ts` – TypeScript interfaces

Example: `PredictionResponse` includes `lipinski: LipinskiStats` and optional `descriptors: ExtendedDescriptors`.

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System status + model readiness |
| `/predict` | POST | Main prediction (SMILES → Active/Inactive) |
| `/train` | POST | Trigger background model retraining |
| `/model/info` | GET | Model metadata + training metrics |
| `/molecule/svg` | POST | Generate 2D structure SVG from SMILES |
| `/molecule/name-to-smiles` | POST | Convert drug name → SMILES via PubChem |

## Chemistry Domain Notes

- **Activity threshold**: 1000 nM (1 μM) – values ≤ threshold = Active
- **Fingerprints**: Morgan fingerprints, radius=2, 2048 bits
- **Lipinski Rule of 5**: MW≤500, LogP≤5, HBD≤5, HBA≤10 – visualized in `LipinskiRadarChart`
- **Training data**: `data/CHEMBL1824_bioactivity.csv` (semicolon-separated)

## Testing Molecules

Active examples (known HER2 inhibitors): Erlotinib, Dacomitinib, Canertinib
Inactive examples: Fluorouracil, Genistein, Bezafibrate

These are defined in `frontend/app/page.tsx` as `ACTIVE_EXAMPLES` and `INACTIVE_EXAMPLES`.
