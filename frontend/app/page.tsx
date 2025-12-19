"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Activity,
  Beaker,
  CheckCircle2,
  AlertCircle,
  Zap,
  TestTube2,
  Dna,
  FlaskConical,
  BarChart3,
  Loader2,
  Atom,
  TrendingUp,
  Shield,
  Search,
  Sparkles,
  TriangleAlert,
  X,
  ZoomIn,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { MoleculeViewer } from "@/components/ui/molecule-viewer";
import { StatusBadge } from "@/components/ui/status-badge";
import { EmptyState } from "@/components/ui/empty-state";
import { LoadingState } from "@/components/ui/loading-state";
import { LipinskiRadarChart } from "@/components/charts/lipinski-radar-chart";
import { DescriptorsBarChart } from "@/components/charts/descriptors-bar-chart";
import { ToggleTheme } from "@/components/ui/toggle-theme";
import { ExportPDFButton } from "@/components/ui/export-pdf-button";
import { api } from "@/lib/api";
import type { PredictionResponse, ModelInfoResponse } from "@/types/api";
import { cn } from "@/lib/utils";

// Compound database from CHEMBL1824 bioactivity dataset
// Activity threshold: IC50 <= 1000 nM = Active
const ALL_COMPOUNDS = [
  // === ACTIVE HER2 INHIBITORS (IC50 < 1000 nM) ===
  { name: "Lapatinib", smiles: "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1", type: "active", ic50: "10 nM" },
  { name: "Afatinib", smiles: "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1", type: "active", ic50: "14 nM" },
  { name: "Neratinib", smiles: "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4ccccn4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C", type: "active", ic50: "59 nM" },
  { name: "Canertinib", smiles: "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1", type: "active", ic50: "7 nM" },
  { name: "Pelitinib", smiles: "CCOc1cc2ncc(C#N)c(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C", type: "active", ic50: "25 nM" },
  { name: "Dacomitinib", smiles: "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1", type: "active", ic50: "6 nM" },
  { name: "Gefitinib", smiles: "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", type: "active", ic50: "90 nM" },
  { name: "Erlotinib", smiles: "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1", type: "active", ic50: "350 nM" },
  { name: "Tucatinib", smiles: "CC1(C)CCN(c2ccc(C(=O)Nc3ccc4c(c3)nn(C)c4N3CCOCC3)nn2)CC1", type: "active", ic50: "8 nM" },
  { name: "Sapitinib", smiles: "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C", type: "active", ic50: "4 nM" },
  { name: "Poziotinib", smiles: "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCC1", type: "active", ic50: "3 nM" },
  { name: "Varlitinib", smiles: "CCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(CC)CC", type: "active", ic50: "18 nM" },
  { name: "Pyrotinib", smiles: "CCOc1cc2ncc(C#N)c(Nc3ccc(OCc4cccnc4)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C", type: "active", ic50: "35 nM" },
  // === INACTIVE COMPOUNDS (IC50 > 10000 nM or Not Active) ===
  { name: "Bezafibrate", smiles: "CC(C)(Oc1ccc(CCNC(=O)c2ccc(Cl)cc2)cc1)C(=O)O", type: "inactive", ic50: "Not Active" },
  { name: "Fluorouracil", smiles: "O=c1[nH]cc(F)c(=O)[nH]1", type: "inactive", ic50: "Not Active" },
  { name: "Genistein", smiles: "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12", type: "inactive", ic50: "Not Active" },
  { name: "Clenbuterol", smiles: "CC(C)(C)NCC(O)c1cc(Cl)c(N)c(Cl)c1", type: "inactive", ic50: "Not Active" },
  { name: "Domperidone", smiles: "O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1", type: "inactive", ic50: "Not Active" },
  { name: "Physostigmine", smiles: "CNC(=O)Oc1ccc2c(c1)[C@]1(C)CCN(C)[C@@H]1N2C", type: "inactive", ic50: "Not Active" },
  { name: "Stavudine", smiles: "Cc1cn([C@H]2C=C[C@@H](CO)O2)c(=O)[nH]c1=O", type: "inactive", ic50: "Not Active" },
  { name: "Ribavirin", smiles: "NC(=O)c1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)n1", type: "inactive", ic50: "Not Active" },
  { name: "Acyclovir", smiles: "Nc1nc2c(ncn2COCCO)c(=O)[nH]1", type: "inactive", ic50: "Not Active" },
  { name: "Nevirapine", smiles: "Cc1ccnc2c1NC(=O)c1cccnc1N2C1CC1", type: "inactive", ic50: "Not Active" },
  { name: "Olanzapine", smiles: "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1", type: "inactive", ic50: "Not Active" },
  { name: "Fluoxetine", smiles: "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", type: "inactive", ic50: "Not Active" },
  { name: "Ondansetron", smiles: "Cc1nccn1CC1CCc2c(c3ccccc3n2C)C1=O", type: "inactive", ic50: "Not Active" },
  { name: "Naproxen", smiles: "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1", type: "inactive", ic50: "Not Active" },
  { name: "Zaleplon", smiles: "CCN(C(C)=O)c1cccc(-c2ccnc3c(C#N)cnn23)c1", type: "inactive", ic50: "Not Active" },
  { name: "Trifluoperazine", smiles: "CN1CCN(CCCN2c3ccccc3Sc3ccc(C(F)(F)F)cc32)CC1", type: "inactive", ic50: "Not Active" },
  { name: "Amprenavir", smiles: "CC(C)CN(C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)O[C@H]1CCOC1)S(=O)(=O)c1ccc(N)cc1", type: "inactive", ic50: "Not Active" },
  { name: "Tirabrutinib", smiles: "CC#CC(=O)N1CC[C@@H](n2c(=O)n(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1", type: "inactive", ic50: "7%" },
  // === MORE ACTIVE HER2 INHIBITORS ===
  { name: "Ibrutinib", smiles: "CC#CC(=O)N1CCC[C@H]1c1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc2n1", type: "active", ic50: "42 nM" },
  { name: "Osimertinib", smiles: "COc1cc(N2CCN(C)CC2)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1", type: "active", ic50: "15 nM" },
  { name: "Mobocertinib", smiles: "COc1cc2ncnc(Nc3ccc(F)c(C#N)c3)c2cc1NC(=O)C=CC(C)(C)N1CCOCC1", type: "active", ic50: "20 nM" },
  { name: "Lazertinib", smiles: "COc1cccc(NC(=O)C=C)c1Nc1nccc(-c2ccc(N3CCN(C)CC3)nc2)n1", type: "active", ic50: "12 nM" },
  { name: "Olmutinib", smiles: "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N1CCOCC1", type: "active", ic50: "9 nM" },
  { name: "Rociletinib", smiles: "COc1cc(N2CCC(N3CCN(C)CC3)CC2)ccc1Nc1ncc(Cl)c(Nc2ccccc2S(=O)(=O)C(C)C)n1", type: "active", ic50: "21 nM" },
  { name: "Naquotinib", smiles: "C=CC(=O)Nc1cccc(Nc2nc(Nc3ccc(N4CCN(C)CC4)cc3OC)ncc2Cl)c1", type: "active", ic50: "16 nM" },
  { name: "Mavelertinib", smiles: "COc1cc(Nc2ncc(Cl)c(Nc3ccccc3S(=O)(=O)C(C)C)n2)ccc1N1CCC(N2CCOCC2)CC1", type: "active", ic50: "28 nM" },
  { name: "Almonertinib", smiles: "C=CC(=O)Nc1cc(Nc2nccc(-c3cnn(C)c3)n2)c(OC)cc1N(C)CCN(C)C", type: "active", ic50: "5 nM" },
  { name: "Furmonertinib", smiles: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2c[nH]c3ncccc23)n1", type: "active", ic50: "8 nM" },
  // === MORE INACTIVE COMPOUNDS ===
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O", type: "inactive", ic50: "Not Active" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1", type: "inactive", ic50: "Not Active" },
  { name: "Metformin", smiles: "CN(C)C(=N)NC(=N)N", type: "inactive", ic50: "Not Active" },
  { name: "Omeprazole", smiles: "COc1ccc2nc(CS(=O)c3ncc(C)c(OC)c3C)[nH]c2c1", type: "inactive", ic50: "Not Active" },
  { name: "Atorvastatin", smiles: "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", type: "inactive", ic50: "Not Active" },
  { name: "Simvastatin", smiles: "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12", type: "inactive", ic50: "Not Active" },
  { name: "Losartan", smiles: "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1", type: "inactive", ic50: "Not Active" },
  { name: "Amlodipine", smiles: "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", type: "inactive", ic50: "Not Active" },
  { name: "Lisinopril", smiles: "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", type: "inactive", ic50: "Not Active" },
  { name: "Gabapentin", smiles: "NCC1(CC(=O)O)CCCCC1", type: "inactive", ic50: "Not Active" },
  { name: "Sertraline", smiles: "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc12", type: "inactive", ic50: "Not Active" },
  { name: "Escitalopram", smiles: "CN(C)CCC[C@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21", type: "inactive", ic50: "Not Active" },
  { name: "Tramadol", smiles: "COc1ccc(C2(O)CCCCC2CN(C)C)cc1", type: "inactive", ic50: "Not Active" },
  { name: "Pantoprazole", smiles: "COc1ccnc(CS(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC", type: "inactive", ic50: "Not Active" },
  { name: "Cetirizine", smiles: "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1", type: "inactive", ic50: "Not Active" },
  { name: "Loratadine", smiles: "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1", type: "inactive", ic50: "Not Active" },
] as const;

export default function Dashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<"online" | "offline" | "checking">("checking");
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const searchRef = useRef<HTMLDivElement>(null);

  // Filter compounds based on search query
  const filteredCompounds = searchQuery.trim()
    ? ALL_COMPOUNDS.filter(
        (c) =>
          c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.smiles.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : ALL_COMPOUNDS;

  const checkHealth = useCallback(async () => {
    try {
      await api.checkHealth();
      setSystemStatus("online");
      if (!modelInfo) {
        try {
          const info = await api.getModelInfo();
          setModelInfo(info);
        } catch {
          // Model info might not be available yet
        }
      }
    } catch {
      setSystemStatus("offline");
    }
  }, [modelInfo]);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowResults(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const runPrediction = async (input: string, moleculeName?: string) => {
    if (!input.trim()) return;
    
    setLoading(true);
    setResult(null);
    setShowResults(false);
    
    try {
      const hasSmilesChars = /[[\]()=#@/\\]/.test(input) || /[a-z]/.test(input);
      const isCapitalizedWord = /^[A-Z][A-Za-z]*$/.test(input);
      const hasSpaces = input.includes(" ");
      
      let smilesString = input;
      
      if (!hasSmilesChars || isCapitalizedWord || hasSpaces) {
        toast.info("Looking up molecule name...");
        try {
          const nameResult = await api.nameToSmiles(input);
          smilesString = nameResult.smiles;
          toast.success(`Found: ${input}`);
        } catch {
          toast.warning("Name lookup failed, trying as SMILES...");
          smilesString = input;
        }
      }
      
      const data = await api.predict(smilesString);
      setResult(data);
      toast.success("Prediction complete!");
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : "An error occurred during prediction";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const handleCompoundClick = (compound: typeof ALL_COMPOUNDS[number]) => {
    setSearchQuery(compound.name);
    setShowResults(false);
    runPrediction(compound.smiles, compound.name);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      runPrediction(searchQuery);
    }
    if (e.key === "Escape") {
      setShowResults(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-11 items-center justify-between px-3 mx-auto max-w-[1600px]">
          <div className="flex items-center gap-2">
            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-teal-500">
              <Dna className="h-4 w-4 text-white" aria-hidden="true" />
            </div>
            <h1 className="text-base font-bold tracking-tight" style={{ fontFamily: 'Belanosima, sans-serif' }}>OncoScope</h1>
          </div>
          <div className="flex items-center gap-2">
            {result && <ExportPDFButton prediction={result} />}
            <StatusBadge
              variant={systemStatus === "online" ? "success" : systemStatus === "offline" ? "error" : "neutral"}
              pulse={systemStatus === "online"}
            >
              {systemStatus === "checking" ? "..." : systemStatus === "online" ? "Online" : "Offline"}
            </StatusBadge>
            <ToggleTheme />
          </div>
        </div>
      </header>

      <main className="flex-1 p-2">
        <div className="max-w-[1600px] mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-2">
            
            {/* Left Column - Input Panel */}
            <div className="lg:col-span-3">
              <Card className="overflow-visible">
                <CardHeader className="p-2 pb-1">
                  <CardTitle className="flex items-center gap-1.5 text-xs">
                    <TestTube2 className="h-3.5 w-3.5 text-teal-500" />
                    Molecular Input
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-2 pt-0 space-y-1.5 overflow-visible">
                  {/* Search Input with Dropdown */}
                  <div ref={searchRef} className="relative">
                    <div className="relative">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                      <Input
                        placeholder="Search compound or enter SMILES..."
                        className="h-9 pl-8 text-xs font-mono"
                        value={searchQuery}
                        onChange={(e) => {
                          setSearchQuery(e.target.value);
                          setShowResults(true);
                        }}
                        onFocus={() => setShowResults(true)}
                        onKeyDown={handleKeyDown}
                      />
                    </div>
                    
                    {/* Dropdown Results */}
                    {showResults && (
                      <div className="absolute z-50 w-full mt-1 bg-popover border border-border rounded-lg shadow-lg max-h-[300px] overflow-y-auto">
                        {filteredCompounds.length === 0 ? (
                          <div className="p-3 text-center text-xs text-muted-foreground">
                            No compounds found. Press Enter to search PubChem.
                          </div>
                        ) : (
                          <div className="p-1">
                            {filteredCompounds.map((compound) => (
                              <button
                                key={compound.name}
                                onClick={() => handleCompoundClick(compound)}
                                className={cn(
                                  "w-full text-left px-2 py-1.5 rounded text-xs hover:bg-muted transition-colors flex items-center justify-between gap-2",
                                )}
                              >
                                <span className="font-medium truncate">{compound.name}</span>
                                <span className="font-mono text-[9px] text-muted-foreground truncate max-w-[140px]">
                                  {compound.smiles.slice(0, 25)}...
                                </span>
                              </button>
                            ))}
                          </div>
                        )}
                        {searchQuery.trim() && (
                          <div className="border-t border-border p-1">
                            <button
                              onClick={() => runPrediction(searchQuery)}
                              className="w-full text-left px-2 py-1.5 rounded text-xs hover:bg-muted transition-colors flex items-center gap-2 text-teal-600 dark:text-teal-400"
                            >
                              <Search className="h-3 w-3" />
                              <span>Search PubChem for &quot;{searchQuery}&quot;</span>
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <Button
                    className="w-full h-8 text-xs bg-teal-500 hover:bg-teal-600 text-white"
                    onClick={() => runPrediction(searchQuery)}
                    disabled={loading || systemStatus !== "online" || !searchQuery.trim()}
                  >
                    {loading ? (
                      <><Loader2 className="mr-1.5 h-3 w-3 animate-spin" />Analyzing...</>
                    ) : (
                      <><Zap className="mr-1.5 h-3 w-3" />Run Prediction</>
                    )}
                  </Button>
                  
                  <p className="text-[9px] text-muted-foreground text-center">
                    Type to search compounds or paste a SMILES string
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Right Column - Results Bento Grid */}
            <div className="lg:col-span-9">
              {loading ? (
                <div className="h-[400px] flex items-center justify-center rounded-lg border border-dashed">
                  <LoadingState variant="card" />
                </div>
              ) : result ? (
                <ResultsBento result={result} />
              ) : (
                <div className="h-[400px] flex items-center justify-center rounded-lg border border-dashed bg-muted/10">
                  <EmptyState
                    icon={<Activity className="h-10 w-10" />}
                    title="Ready to Analyze"
                    description="Search for a compound or enter a SMILES string"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="border-t border-border">
        <p className="text-[9px] text-muted-foreground text-center py-1.5">
          OncoScope — AI-Powered Drug Discovery • Research Use Only
        </p>
      </footer>
    </div>
  );
}

/**
 * ResultsBento - Compact bento grid layout for results
 */
function ResultsBento({ result }: { result: PredictionResponse }) {
  const isActive = result.prediction === "Active";

  const [showMoleculeModal, setShowMoleculeModal] = useState(false);

  const lipinskiStats = [
    { label: "MW", value: result.lipinski.MW, unit: "Da", threshold: 500 },
    { label: "LogP", value: result.lipinski.LogP, unit: "", threshold: 5 },
    { label: "HBD", value: result.lipinski.NumHDonors, unit: "", threshold: 5 },
    { label: "HBA", value: result.lipinski.NumHAcceptors, unit: "", threshold: 10 },
  ];

  const extendedStats = result.descriptors ? [
    { label: "TPSA", value: result.descriptors.TPSA, threshold: 140 },
    { label: "RotB", value: result.descriptors.NumRotatableBonds, threshold: 10 },
    { label: "Rings", value: result.descriptors.NumAromaticRings, threshold: 4 },
    { label: "Heavy", value: result.descriptors.NumHeavyAtoms, threshold: 999 },
  ] : [];

  return (
    <>
    {/* Molecule Modal */}
    {showMoleculeModal && (
      <div 
        className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
        onClick={() => setShowMoleculeModal(false)}
      >
        <div 
          className="bg-background rounded-xl p-4 max-w-2xl w-full shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Atom className="h-5 w-5 text-teal-500" />
              <h3 className="text-lg font-semibold">2D Molecular Structure</h3>
            </div>
            <button
              onClick={() => setShowMoleculeModal(false)}
              className="p-1 rounded-lg hover:bg-muted transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="bg-white rounded-lg p-4 flex items-center justify-center">
            <MoleculeViewer smiles={result.smiles} width={500} height={400} />
          </div>
          <div className="mt-3 p-2 bg-muted/50 rounded-lg">
            <p className="text-xs text-muted-foreground mb-1">SMILES</p>
            <p className="text-sm font-mono break-all">{result.smiles}</p>
          </div>
          {result.descriptors?.MolecularFormula && (
            <div className="mt-2 flex items-center gap-2">
              <Beaker className="h-4 w-4 text-teal-500" />
              <span className="text-sm">Formula: <strong className="font-mono">{result.descriptors.MolecularFormula}</strong></span>
            </div>
          )}
        </div>
      </div>
    )}
    
    <div className="grid grid-cols-4 gap-1.5 auto-rows-[minmax(80px,auto)]">
      {/* Prediction Result - Col span 2 */}
      <Card className={cn(
        "col-span-2 row-span-2 p-3 border-2",
        isActive 
          ? "border-teal-500/40 bg-teal-500/5" 
          : "border-rose-500/40 bg-rose-500/5"
      )}>
        <div className="h-full flex flex-col justify-between">
          <div className="flex items-start gap-3">
            <div className={cn(
              "flex items-center justify-center w-12 h-12 rounded-xl shrink-0",
              isActive ? "bg-teal-500/20 text-teal-500" : "bg-rose-500/20 text-rose-500"
            )}>
              {isActive ? <CheckCircle2 className="h-7 w-7" /> : <AlertCircle className="h-7 w-7" />}
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">HER2 Bioactivity</p>
              <p className={cn(
                "text-3xl font-bold",
                isActive ? "text-teal-600 dark:text-teal-400" : "text-rose-600 dark:text-rose-400"
              )}>
                {result.prediction}
              </p>
            </div>
          </div>
          <div className="space-y-1.5 mt-2">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground flex items-center gap-1"><Shield className="h-3 w-3" />Confidence</span>
              <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <Progress value={result.confidence * 100} className={cn("h-1.5", isActive ? "[&>div]:bg-teal-500" : "[&>div]:bg-rose-500")} />
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground flex items-center gap-1"><TrendingUp className="h-3 w-3" />Active Prob</span>
              <span className="font-semibold text-teal-600 dark:text-teal-400">{(result.probability * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </Card>

      {/* 2D Molecule Structure - Col span 2 */}
      <Card className="col-span-2 row-span-2 p-2 flex flex-col">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5">
            <Atom className="h-3.5 w-3.5 text-teal-500" />
            <span className="text-xs font-medium">2D Structure</span>
          </div>
          <span className="text-[9px] text-muted-foreground flex items-center gap-1">
            <ZoomIn className="h-3 w-3" /> Click to enlarge
          </span>
        </div>
        <button
          onClick={() => setShowMoleculeModal(true)}
          className="flex-1 flex items-center justify-center bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors cursor-zoom-in"
        >
          <MoleculeViewer smiles={result.smiles} width={200} height={140} />
        </button>
        <p className="text-[9px] font-mono text-muted-foreground mt-1 truncate">{result.smiles}</p>
      </Card>

      {/* Lipinski Radar Chart */}
      <Card className="col-span-2 p-2">
        <div className="flex items-center gap-1.5 mb-1">
          <FlaskConical className="h-3.5 w-3.5 text-teal-500" />
          <span className="text-xs font-medium">Drug-Likeness</span>
        </div>
        <div className="h-[100px]">
          <LipinskiRadarChart lipinski={result.lipinski} compact />
        </div>
      </Card>

      {/* Descriptors Bar Chart */}
      {result.descriptors && (
        <Card className="col-span-2 p-2">
          <div className="flex items-center gap-1.5 mb-1">
            <BarChart3 className="h-3.5 w-3.5 text-teal-500" />
            <span className="text-xs font-medium">Properties</span>
          </div>
          <div className="h-[100px]">
            <DescriptorsBarChart descriptors={result.descriptors} compact />
          </div>
        </Card>
      )}

      {/* QED Score Card */}
      {result.drug_likeness && (
        <Card className={cn(
          "col-span-2 p-2 border-2",
          result.drug_likeness.qed >= 0.67 ? "border-teal-500/40 bg-teal-500/5" :
          result.drug_likeness.qed >= 0.49 ? "border-blue-500/40 bg-blue-500/5" :
          result.drug_likeness.qed >= 0.34 ? "border-amber-500/40 bg-amber-500/5" :
          "border-rose-500/40 bg-rose-500/5"
        )}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Sparkles className={cn(
                "h-3.5 w-3.5",
                result.drug_likeness.qed >= 0.67 ? "text-teal-500" :
                result.drug_likeness.qed >= 0.49 ? "text-blue-500" :
                result.drug_likeness.qed >= 0.34 ? "text-amber-500" :
                "text-rose-500"
              )} />
              <span className="text-xs font-medium">QED Score</span>
            </div>
            <span className={cn(
              "text-[10px] px-1.5 py-0.5 rounded font-medium",
              result.drug_likeness.qed >= 0.67 ? "bg-teal-500/10 text-teal-600" :
              result.drug_likeness.qed >= 0.49 ? "bg-blue-500/10 text-blue-600" :
              result.drug_likeness.qed >= 0.34 ? "bg-amber-500/10 text-amber-600" :
              "bg-rose-500/10 text-rose-600"
            )}>
              {result.drug_likeness.qed_category}
            </span>
          </div>
          <div className="flex items-end gap-2">
            <span className={cn(
              "text-3xl font-bold tabular-nums",
              result.drug_likeness.qed >= 0.67 ? "text-teal-600 dark:text-teal-400" :
              result.drug_likeness.qed >= 0.49 ? "text-blue-600 dark:text-blue-400" :
              result.drug_likeness.qed >= 0.34 ? "text-amber-600 dark:text-amber-400" :
              "text-rose-600 dark:text-rose-400"
            )}>
              {result.drug_likeness.qed.toFixed(3)}
            </span>
            <span className="text-[10px] text-muted-foreground mb-1">/ 1.0</span>
          </div>
          <Progress 
            value={result.drug_likeness.qed * 100} 
            className={cn(
              "h-1.5 mt-2",
              result.drug_likeness.qed >= 0.67 ? "[&>div]:bg-teal-500" :
              result.drug_likeness.qed >= 0.49 ? "[&>div]:bg-blue-500" :
              result.drug_likeness.qed >= 0.34 ? "[&>div]:bg-amber-500" :
              "[&>div]:bg-rose-500"
            )} 
          />
        </Card>
      )}

      {/* Drug-Likeness Compliance Card */}
      {result.drug_likeness && (
        <Card className="col-span-2 p-2">
          <div className="flex items-center gap-1.5 mb-2">
            <Shield className="h-3.5 w-3.5 text-teal-500" />
            <span className="text-xs font-medium">Compliance Rules</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5">
            <div className={cn(
              "p-1.5 rounded text-center",
              result.drug_likeness.lipinski_compliant 
                ? "bg-teal-500/10" 
                : "bg-rose-500/10"
            )}>
              <p className="text-[9px] text-muted-foreground">Lipinski</p>
              <p className={cn(
                "text-xs font-semibold",
                result.drug_likeness.lipinski_compliant 
                  ? "text-teal-600 dark:text-teal-400" 
                  : "text-rose-600 dark:text-rose-400"
              )}>
                {result.drug_likeness.lipinski_compliant ? "Pass" : `${result.drug_likeness.lipinski_violations} viol.`}
              </p>
            </div>
            <div className={cn(
              "p-1.5 rounded text-center",
              result.drug_likeness.veber_compliant 
                ? "bg-teal-500/10" 
                : "bg-rose-500/10"
            )}>
              <p className="text-[9px] text-muted-foreground">Veber</p>
              <p className={cn(
                "text-xs font-semibold",
                result.drug_likeness.veber_compliant 
                  ? "text-teal-600 dark:text-teal-400" 
                  : "text-rose-600 dark:text-rose-400"
              )}>
                {result.drug_likeness.veber_compliant ? "Pass" : "Fail"}
              </p>
            </div>
            <div className={cn(
              "p-1.5 rounded text-center",
              result.drug_likeness.pains_count === 0 
                ? "bg-teal-500/10" 
                : "bg-rose-500/10"
            )}>
              <p className="text-[9px] text-muted-foreground">PAINS</p>
              <p className={cn(
                "text-xs font-semibold",
                result.drug_likeness.pains_count === 0 
                  ? "text-teal-600 dark:text-teal-400" 
                  : "text-rose-600 dark:text-rose-400"
              )}>
                {result.drug_likeness.pains_count === 0 ? "Clear" : `${result.drug_likeness.pains_count} alert${result.drug_likeness.pains_count > 1 ? "s" : ""}`}
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* PAINS Alerts Warning */}
      {result.drug_likeness && result.drug_likeness.pains_count > 0 && (
        <Card className="col-span-4 p-2 border-amber-500/40 bg-amber-500/5">
          <div className="flex items-start gap-2">
            <TriangleAlert className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-medium text-amber-600 dark:text-amber-400">PAINS Alerts Detected</p>
              <p className="text-[10px] text-muted-foreground mt-0.5">
                {result.drug_likeness.pains_alerts.join(", ")}
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Lipinski Stats Row */}
      {lipinskiStats.map((stat) => {
        const isOk = Number(stat.value) <= stat.threshold;
        return (
          <Card key={stat.label} className={cn(
            "p-2 flex flex-col justify-between",
            !isOk && "border-rose-500/30 bg-rose-500/5"
          )}>
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-muted-foreground uppercase">{stat.label}</span>
              <span className={cn(
                "text-[8px] px-1 py-0.5 rounded font-medium",
                isOk ? "bg-teal-500/10 text-teal-600" : "bg-rose-500/10 text-rose-600"
              )}>
                {isOk ? "OK" : "HIGH"}
              </span>
            </div>
            <div className="flex items-baseline gap-0.5">
              <span className="text-xl font-bold tabular-nums">
                {typeof stat.value === 'number' ? stat.value.toFixed(stat.unit ? 1 : 0) : stat.value}
              </span>
              {stat.unit && <span className="text-[9px] text-muted-foreground">{stat.unit}</span>}
            </div>
          </Card>
        );
      })}

      {/* Extended Stats */}
      {extendedStats.map((stat) => {
        const isOk = Number(stat.value) <= stat.threshold;
        return (
          <Card key={stat.label} className={cn(
            "p-2 flex flex-col justify-between",
            !isOk && stat.threshold !== 999 && "border-rose-500/30 bg-rose-500/5"
          )}>
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-muted-foreground uppercase">{stat.label}</span>
              {stat.threshold !== 999 && (
                <span className={cn(
                  "text-[8px] px-1 py-0.5 rounded font-medium",
                  isOk ? "bg-teal-500/10 text-teal-600" : "bg-rose-500/10 text-rose-600"
                )}>
                  {isOk ? "OK" : "HIGH"}
                </span>
              )}
            </div>
            <span className="text-xl font-bold tabular-nums">
              {typeof stat.value === 'number' ? stat.value.toFixed(Number.isInteger(stat.value) ? 0 : 1) : stat.value}
            </span>
          </Card>
        );
      })}

      {/* Molecular Formula */}
      {result.descriptors?.MolecularFormula && (
        <Card className="col-span-4 p-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Beaker className="h-4 w-4 text-teal-500" />
            <span className="text-xs font-medium">Molecular Formula</span>
          </div>
          <span className="font-mono text-sm font-semibold">{result.descriptors.MolecularFormula}</span>
        </Card>
      )}
    </div>
    </>
  );
}
