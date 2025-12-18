"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Activity,
  Beaker,
  CheckCircle2,
  AlertCircle,
  Zap,
  RefreshCcw,
  TestTube2,
  Dna,
  Server,
  FlaskConical,
  BarChart3,
  Loader2,
  Atom,
  Target,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { MoleculeViewer } from "@/components/ui/molecule-viewer";
import { StatusBadge } from "@/components/ui/status-badge";
import { EmptyState } from "@/components/ui/empty-state";
import { LoadingState } from "@/components/ui/loading-state";
import { PropertyBadge } from "@/components/ui/property-badge";
import { LipinskiRadarChart } from "@/components/charts/lipinski-radar-chart";
import { DescriptorsBarChart } from "@/components/charts/descriptors-bar-chart";
import { ToggleTheme } from "@/components/ui/toggle-theme";
import { api } from "@/lib/api";
import type { PredictionResponse, ModelInfoResponse } from "@/types/api";

// Real compounds from HER2 training data (from CHEMBL1824_bioactivity.csv)
const ACTIVE_EXAMPLES = [
  { name: "Sorafenib", smiles: "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1" },
  { name: "Dacomitinib", smiles: "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1" },
  { name: "Erlotinib", smiles: "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1" },
  { name: "Pelitinib", smiles: "CCOc1cc2ncc(C#N)c(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C" },
  { name: "Canertinib", smiles: "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1" },
] as const;

const INACTIVE_EXAMPLES = [
  { name: "Bezafibrate", smiles: "CC(C)(Oc1ccc(CCNC(=O)c2ccc(Cl)cc2)cc1)C(=O)O" },
  { name: "Fluorouracil", smiles: "O=c1[nH]cc(F)c(=O)[nH]1" },
  { name: "Genistein", smiles: "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12" },
  { name: "Clenbuterol", smiles: "CC(C)(C)NCC(O)c1cc(Cl)c(N)c(Cl)c1" },
  { name: "Domperidone", smiles: "O=c1[nH]c2ccccc2n1CCCN1CCC(n2c(=O)[nH]c3cc(Cl)ccc32)CC1" },
] as const;

export default function Dashboard() {
  const [smiles, setSmiles] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<"online" | "offline" | "checking">("checking");
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [isRetraining, setIsRetraining] = useState(false);

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

  const handlePredict = async () => {
    const input = smiles.trim();
    if (!input) {
      toast.error("Please enter a SMILES string or molecule name");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      // Check if input looks like a SMILES - must contain typical SMILES special characters
      // SMILES typically have: parentheses, brackets, =, #, numbers, or lowercase letters (aromatic)
      const hasSmilesChars = /[[\]()=#@/\\]/.test(input) || /[a-z]/.test(input);
      const isCapitalizedWord = /^[A-Z][A-Za-z]*$/.test(input); // Like "Aspirin", "Erlotinib"
      const hasSpaces = input.includes(" ");
      
      let smilesString = input;
      
      // If it looks like a molecule name (no SMILES special chars, or is a capitalized word)
      if (!hasSmilesChars || isCapitalizedWord || hasSpaces) {
        toast.info("Looking up molecule name...");
        try {
          const nameResult = await api.nameToSmiles(input);
          smilesString = nameResult.smiles;
          toast.success(`Found: ${input}`);
        } catch (nameError: any) {
          // If name lookup fails, try as SMILES anyway
          toast.warning("Name lookup failed, trying as SMILES...");
          smilesString = input;
        }
      }
      
      const data = await api.predict(smilesString);
      setResult(data);
      toast.success("Prediction complete!");
    } catch (error: any) {
      toast.error(error.message || "An error occurred during prediction");
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    if (isRetraining) return;
    setIsRetraining(true);
    try {
      const data = await api.train();
      toast.info(data.message || "Training started");
    } catch {
      toast.error("Failed to trigger training");
    } finally {
      setIsRetraining(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handlePredict();
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-12 items-center justify-between px-4 sm:px-6 lg:px-8 mx-auto max-w-7xl">
          <div className="flex items-center gap-2">
            <Dna className="h-5 w-5 text-primary" aria-hidden="true" />
            <h1 className="text-lg font-semibold tracking-tight">OncoScope</h1>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge
              variant={systemStatus === "online" ? "success" : systemStatus === "offline" ? "error" : "neutral"}
              pulse={systemStatus === "online"}
              aria-live="polite"
            >
              {systemStatus === "checking" ? "..." : systemStatus === "online" ? "Online" : "Offline"}
            </StatusBadge>
            <ToggleTheme />
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-7xl px-2 py-2 flex-1">
        {/* Two Column Layout: Input + Results */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-2">
          {/* Input Panel - Fixed Width */}
          <div className="lg:col-span-3">
            <Card className="h-full">
              <CardHeader className="p-3 pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <TestTube2 className="h-4 w-4 text-primary" aria-hidden="true" />
                  Molecular Input
                </CardTitle>
              </CardHeader>
              <CardContent className="p-3 pt-0 space-y-2">
                <Textarea
                  id="smiles-input"
                  placeholder="Enter SMILES string or molecule name (e.g., Aspirin, Lapatinib)"
                  className="min-h-[100px] font-mono text-xs resize-none"
                  value={smiles}
                  onChange={(e) => setSmiles(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
                <div className="grid grid-cols-2 gap-1">
                  <Select onValueChange={(val) => setSmiles(val)}>
                    <SelectTrigger className="h-7 text-xs">
                      <SelectValue placeholder="Active" />
                    </SelectTrigger>
                    <SelectContent>
                      {ACTIVE_EXAMPLES.map((mol) => (
                        <SelectItem key={mol.name} value={mol.smiles} className="text-xs">
                          {mol.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select onValueChange={(val) => setSmiles(val)}>
                    <SelectTrigger className="h-7 text-xs">
                      <SelectValue placeholder="Inactive" />
                    </SelectTrigger>
                    <SelectContent>
                      {INACTIVE_EXAMPLES.map((mol) => (
                        <SelectItem key={mol.name} value={mol.smiles} className="text-xs">
                          {mol.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
              <CardFooter className="p-3 pt-0">
                <Button
                  className="w-full"
                  size="sm"
                  onClick={handlePredict}
                  disabled={loading || systemStatus !== "online"}
                  aria-busy={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-3.5 w-3.5" aria-hidden="true" />
                      Run Prediction
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>
          </div>

          {/* Results Panel - Bento Grid */}
          <div className="lg:col-span-9" aria-live="polite">
            {loading ? (
              <div className="h-full min-h-[350px] flex items-center justify-center">
                <LoadingState variant="card" />
              </div>
            ) : result ? (
              <ResultsBento result={result} />
            ) : (
              <div className="h-full min-h-[350px] flex items-center justify-center rounded-lg border border-dashed border-border bg-muted/20">
                <EmptyState
                  icon={<Activity className="h-10 w-10" />}
                  title="Ready to Analyze"
                  description="Enter a SMILES string to see results"
                />
              </div>
            )}
          </div>
        </div>

        {/* Admin Section - Collapsible */}
        <Accordion type="single" collapsible className="w-full mt-2">
          <AccordionItem value="diagnostics" className="border rounded-lg px-3">
            <AccordionTrigger className="text-xs text-muted-foreground hover:text-foreground hover:no-underline py-2">
              <div className="flex items-center gap-2">
                <Server className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Model Administration</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="pb-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div>
                  <span className="text-muted-foreground">Status:</span>
                  <span className="ml-1 font-medium capitalize">{modelInfo?.status || "Unknown"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Backend:</span>
                  <span className="ml-1 font-medium">RandomForest</span>
                </div>
                {modelInfo?.metadata?.metrics && (
                  <>
                    <div>
                      <span className="text-muted-foreground">Accuracy:</span>
                      <span className="ml-1 font-medium">{(modelInfo.metadata.metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">F1:</span>
                      <span className="ml-1 font-medium">{(modelInfo.metadata.metrics.f1_score * 100).toFixed(1)}%</span>
                    </div>
                  </>
                )}
                <div className="col-span-2 sm:col-span-4 pt-2">
                  <Button variant="destructive" size="sm" className="h-7 text-xs" onClick={handleRetrain} disabled={isRetraining}>
                    {isRetraining ? <Loader2 className="mr-1.5 h-3 w-3 animate-spin" /> : <RefreshCcw className="mr-1.5 h-3 w-3" />}
                    {isRetraining ? "Training..." : "Retrain Model"}
                  </Button>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="container mx-auto max-w-7xl px-4 py-2">
          <p className="text-[10px] text-muted-foreground text-center">
            OncoScope — AI-Powered Drug Discovery • Research Use Only
          </p>
        </div>
      </footer>
    </div>
  );
}

// Bento Grid Results Display
function ResultsBento({ result }: { result: PredictionResponse }) {
  const isActive = result.prediction === "Active";

  // Lipinski summary
  const lipinskiData = [
    { label: "MW", value: result.lipinski.MW, unit: "Da", threshold: 500 },
    { label: "LogP", value: result.lipinski.LogP, unit: "", threshold: 5 },
    { label: "HBD", value: result.lipinski.NumHDonors, unit: "", threshold: 5 },
    { label: "HBA", value: result.lipinski.NumHAcceptors, unit: "", threshold: 10 },
  ];

  return (
    <BentoGrid className="md:grid-cols-4 md:auto-rows-[8.5rem] gap-1">
      {/* Prediction Result - Spans 2 cols */}
      <BentoGridItem
        className={`md:col-span-2 ${
          isActive
            ? "border-emerald-500/40 bg-emerald-500/5"
            : "border-red-500/40 bg-red-500/5"
        }`}
        header={
          <div className="flex items-center justify-between h-full">
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                {isActive ? (
                  <CheckCircle2 className="h-6 w-6 text-emerald-500" />
                ) : (
                  <AlertCircle className="h-6 w-6 text-red-500" />
                )}
                <span className={`text-base font-bold ${isActive ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                  {result.prediction.toUpperCase()}
                </span>
              </div>
              <div className="space-y-0.5">
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Confidence:</span>
                  <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">HER2 Prob:</span>
                  <span className="font-semibold text-primary">{(result.probability * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.probability * 100} className="h-1.5 w-32" />
              </div>
            </div>
          </div>
        }
        title={<span className="text-xs">Bioactivity Prediction</span>}
        icon={<Target className="h-3.5 w-3.5 text-primary" />}
      />

      {/* Molecule Viewer - Spans 2 cols */}
      <BentoGridItem
        className="md:col-span-2"
        header={
          <div className="flex items-center justify-center h-full">
            <MoleculeViewer smiles={result.smiles} width={160} height={100} />
          </div>
        }
        title={<span className="text-xs truncate max-w-[200px]">{result.smiles.slice(0, 30)}...</span>}
        icon={<Atom className="h-3.5 w-3.5 text-primary" />}
      />

      {/* Lipinski Radar Chart */}
      <BentoGridItem
        className="md:col-span-2"
        header={
          <div className="w-full h-full">
            <LipinskiRadarChart lipinski={result.lipinski} compact />
          </div>
        }
        title={<span className="text-xs">Drug-Likeness</span>}
        icon={<FlaskConical className="h-3.5 w-3.5 text-primary" />}
      />

      {/* Descriptors Bar Chart */}
      {result.descriptors && (
        <BentoGridItem
          className="md:col-span-2"
          header={
            <div className="w-full h-full">
              <DescriptorsBarChart descriptors={result.descriptors} compact />
            </div>
          }
          title={<span className="text-xs">Molecular Properties</span>}
          icon={<BarChart3 className="h-3.5 w-3.5 text-primary" />}
        />
      )}

      {/* Lipinski Properties Summary - Spans 4 cols */}
      <BentoGridItem
        className="md:col-span-4"
        header={
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-1 w-full">
            {lipinskiData.map((item) => (
              <div key={item.label} className="flex flex-col items-center p-1.5 rounded bg-muted/50">
                <span className="text-[9px] text-muted-foreground uppercase">{item.label}</span>
                <span className="text-xs font-semibold tabular-nums">{typeof item.value === 'number' ? item.value.toFixed(1) : item.value}</span>
                <PropertyBadge 
                  status={Number(item.value) <= item.threshold ? "optimal" : "warning"} 
                  className="text-[8px] px-1 py-0"
                  showIcon={false}
                >
                  {Number(item.value) <= item.threshold ? "OK" : "High"}
                </PropertyBadge>
              </div>
            ))}
            {result.descriptors && (
              <>
                <div className="flex flex-col items-center p-1.5 rounded bg-muted/50">
                  <span className="text-[9px] text-muted-foreground uppercase">TPSA</span>
                  <span className="text-xs font-semibold tabular-nums">{result.descriptors.TPSA.toFixed(1)}</span>
                  <PropertyBadge 
                    status={result.descriptors.TPSA <= 140 ? "optimal" : "warning"}
                    className="text-[8px] px-1 py-0"
                    showIcon={false}
                  >
                    {result.descriptors.TPSA <= 140 ? "OK" : "High"}
                  </PropertyBadge>
                </div>
                <div className="flex flex-col items-center p-1.5 rounded bg-muted/50">
                  <span className="text-[9px] text-muted-foreground uppercase">RotB</span>
                  <span className="text-xs font-semibold tabular-nums">{result.descriptors.NumRotatableBonds}</span>
                  <PropertyBadge 
                    status={result.descriptors.NumRotatableBonds <= 10 ? "optimal" : "warning"}
                    className="text-[8px] px-1 py-0"
                    showIcon={false}
                  >
                    {result.descriptors.NumRotatableBonds <= 10 ? "OK" : "High"}
                  </PropertyBadge>
                </div>
                <div className="flex flex-col items-center p-1.5 rounded bg-muted/50">
                  <span className="text-[9px] text-muted-foreground uppercase">Rings</span>
                  <span className="text-xs font-semibold tabular-nums">{result.descriptors.NumAromaticRings}</span>
                  <PropertyBadge 
                    status={result.descriptors.NumAromaticRings <= 3 ? "optimal" : "warning"}
                    className="text-[8px] px-1 py-0"
                    showIcon={false}
                  >
                    {result.descriptors.NumAromaticRings <= 3 ? "OK" : "High"}
                  </PropertyBadge>
                </div>
                <div className="flex flex-col items-center p-1.5 rounded bg-muted/50">
                  <span className="text-[9px] text-muted-foreground uppercase">Heavy</span>
                  <span className="text-xs font-semibold tabular-nums">{result.descriptors.NumHeavyAtoms}</span>
                  <PropertyBadge status="info" className="text-[8px] px-1 py-0" showIcon={false}>
                    Info
                  </PropertyBadge>
                </div>
              </>
            )}
          </div>
        }
        title={<span className="text-xs">Lipinski Rule of Five & Descriptors</span>}
        icon={<Beaker className="h-3.5 w-3.5 text-primary" />}
        description={result.descriptors?.MolecularFormula ? `Formula: ${result.descriptors.MolecularFormula}` : undefined}
      />
    </BentoGrid>
  );
}
