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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { MoleculeViewer } from "@/components/ui/molecule-viewer";
import { StatusBadge } from "@/components/ui/status-badge";
import { EmptyState } from "@/components/ui/empty-state";
import { LoadingState } from "@/components/ui/loading-state";
import { PropertyBadge } from "@/components/ui/property-badge";
import { LipinskiRadarChart } from "@/components/charts/lipinski-radar-chart";
import { DescriptorsBarChart } from "@/components/charts/descriptors-bar-chart";
import { ThemeToggle } from "@/components/theme-toggle";
import { api } from "@/lib/api";
import type { PredictionResponse, ModelInfoResponse } from "@/types/api";

// Real compounds from HER2 training data
const EXAMPLE_MOLECULES = {
  active: "CN1CCN(CCC#CC(=O)Nc2cc3c(Nc4ccc(F)c(Cl)c4)ncnc3cn2)CC1",
  inactive: "N#C/C(=C\\c1ccc(O)c(O)c1)C(=O)c1ccc(F)cc1",
} as const;

export default function Dashboard() {
  const [smiles, setSmiles] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<"online" | "offline" | "checking">("checking");
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [isRetraining, setIsRetraining] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const data = await api.checkHealth();
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
    const trimmedSmiles = smiles.trim();
    if (!trimmedSmiles) {
      toast.error("Please enter a valid SMILES string");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const data = await api.predict(trimmedSmiles);
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

  const getPropertyStatus = (value: number, threshold: number): "optimal" | "warning" => {
    return value <= threshold ? "optimal" : "warning";
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 sm:h-16 items-center justify-between px-4 sm:px-6 lg:px-8 mx-auto max-w-7xl">
          <div className="flex items-center gap-2">
            <Dna className="h-5 w-5 sm:h-6 sm:w-6 text-primary" aria-hidden="true" />
            <h1 className="text-lg sm:text-xl font-semibold tracking-tight">OncoScope</h1>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <span className="text-xs sm:text-sm text-muted-foreground hidden sm:inline">Status:</span>
            <StatusBadge
              variant={systemStatus === "online" ? "success" : systemStatus === "offline" ? "error" : "neutral"}
              pulse={systemStatus === "online"}
              aria-live="polite"
            >
              {systemStatus === "checking" ? "Connecting..." : systemStatus === "online" ? "Online" : "Offline"}
            </StatusBadge>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 sm:space-y-8 flex-1">
        {/* Hero */}
        <div className="text-center space-y-2 pb-2">
          <h2 className="text-xl sm:text-2xl font-semibold tracking-tight">AI-Powered HER2 Bioactivity Prediction</h2>
          <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto">
            Analyze molecular structures for potential HER2 inhibitory activity using machine learning
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Input Section */}
          <section aria-labelledby="input-heading">
            <Card>
              <CardHeader>
                <CardTitle id="input-heading" className="flex items-center gap-2 text-base">
                  <TestTube2 className="h-5 w-5 text-primary" aria-hidden="true" />
                  Molecular Input
                </CardTitle>
                <CardDescription>Enter a SMILES string to analyze its potential as a HER2 inhibitor</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="smiles-input" className="sr-only">SMILES string input</label>
                  <Textarea
                    id="smiles-input"
                    placeholder="Enter SMILES string e.g., CN1CCN(CCC#CC(=O)Nc2cc3c..."
                    className="min-h-[120px] sm:min-h-[150px] font-mono text-sm resize-none"
                    value={smiles}
                    onChange={(e) => setSmiles(e.target.value)}
                    onKeyDown={handleKeyDown}
                    aria-describedby="smiles-hint"
                  />
                  <p id="smiles-hint" className="text-xs text-muted-foreground">
                    Press <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px] font-mono">⌘+Enter</kbd> to run prediction
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button variant="outline" size="sm" onClick={() => setSmiles(EXAMPLE_MOLECULES.active)}>
                    Load Active Example
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => setSmiles(EXAMPLE_MOLECULES.inactive)}>
                    Load Inactive Example
                  </Button>
                </div>
              </CardContent>
              <CardFooter>
                <Button
                  className="w-full"
                  size="lg"
                  onClick={handlePredict}
                  disabled={loading || systemStatus !== "online"}
                  aria-busy={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                      Analyzing Structure...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-4 w-4" aria-hidden="true" />
                      Run Prediction
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>
          </section>

          {/* Results Section */}
          <section aria-labelledby="results-heading" aria-live="polite">
            <h2 id="results-heading" className="sr-only">Prediction Results</h2>
            {loading ? (
              <LoadingState variant="full" />
            ) : result ? (
              <ResultsDisplay result={result} getPropertyStatus={getPropertyStatus} />
            ) : (
              <EmptyState
                icon={<Activity className="h-12 w-12" />}
                title="Ready to Analyze"
                description="Enter a SMILES string and run a prediction to see the bioactivity analysis results."
                className="h-full min-h-[300px]"
              />
            )}
          </section>
        </div>

        {/* Admin Section */}
        <section className="pt-6 border-t border-border" aria-labelledby="admin-heading">
          <h2 id="admin-heading" className="sr-only">Administration</h2>
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="diagnostics" className="border-b-0">
              <AccordionTrigger className="text-muted-foreground hover:text-foreground hover:no-underline py-3">
                <div className="flex items-center gap-2">
                  <Server className="h-4 w-4" aria-hidden="true" />
                  <span>Model Diagnostics & Administration</span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="pt-4">
                <Card className="bg-muted/30">
                  <CardContent className="p-4 sm:p-6 grid gap-6 sm:grid-cols-2">
                    <div className="space-y-4">
                      <h3 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">Current Model Stats</h3>
                      <dl className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <dt className="text-muted-foreground text-xs">Status</dt>
                          <dd className="font-medium capitalize">{modelInfo?.status || "Unknown"}</dd>
                        </div>
                        <div>
                          <dt className="text-muted-foreground text-xs">Backend</dt>
                          <dd className="font-medium">RandomForest</dd>
                        </div>
                        <div className="col-span-2">
                          <dt className="text-muted-foreground text-xs">Last Trained</dt>
                          <dd className="font-medium truncate">
                            {modelInfo?.metadata?.training_date
                              ? new Date(modelInfo.metadata.training_date).toLocaleString()
                              : "N/A"}
                          </dd>
                        </div>
                        {modelInfo?.metadata?.metrics && (
                          <>
                            <div>
                              <dt className="text-muted-foreground text-xs">Accuracy</dt>
                              <dd className="font-medium">{(modelInfo.metadata.metrics.accuracy * 100).toFixed(1)}%</dd>
                            </div>
                            <div>
                              <dt className="text-muted-foreground text-xs">F1 Score</dt>
                              <dd className="font-medium">{(modelInfo.metadata.metrics.f1_score * 100).toFixed(1)}%</dd>
                            </div>
                          </>
                        )}
                      </dl>
                    </div>
                    <div className="flex flex-col justify-end items-start sm:items-end gap-3 border-t sm:border-t-0 sm:border-l border-border pt-4 sm:pt-0 sm:pl-6">
                      <p className="text-xs text-muted-foreground sm:text-right">
                        Trigger a full model retrain using the latest dataset
                      </p>
                      <Button variant="destructive" onClick={handleRetrain} disabled={isRetraining}>
                        {isRetraining ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Training...
                          </>
                        ) : (
                          <>
                            <RefreshCcw className="mr-2 h-4 w-4" />
                            Retrain Model
                          </>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-xs text-muted-foreground text-center">
            OncoScope — AI-Powered Drug Discovery Platform • For research purposes only
          </p>
        </div>
      </footer>
    </div>
  );
}

// Separated Results component for better organization
interface ResultsDisplayProps {
  result: PredictionResponse;
  getPropertyStatus: (value: number, threshold: number) => "optimal" | "warning";
}

function ResultsDisplay({ result, getPropertyStatus }: ResultsDisplayProps) {
  const isActive = result.prediction === "Active";

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Verdict Card */}
      <Card
        className={`border-2 ${
          isActive
            ? "border-emerald-500/50 bg-emerald-500/5 dark:bg-emerald-500/10"
            : "border-red-500/50 bg-red-500/5 dark:bg-red-500/10"
        }`}
      >
        <CardContent className="pt-6 p-4 sm:p-6">
          <div className="flex flex-col sm:flex-row items-center gap-4 sm:gap-6">
            <div className="flex-shrink-0">
              <MoleculeViewer smiles={result.smiles} width={160} height={160} />
            </div>
            <div className="flex flex-col items-center sm:items-start text-center sm:text-left flex-1 space-y-3">
              <div className="flex items-center gap-3">
                {isActive ? (
                  <CheckCircle2 className="h-8 w-8 sm:h-10 sm:w-10 text-emerald-500" aria-hidden="true" />
                ) : (
                  <AlertCircle className="h-8 w-8 sm:h-10 sm:w-10 text-red-500" aria-hidden="true" />
                )}
                <h3
                  className={`text-xl sm:text-2xl font-bold tracking-tight ${
                    isActive ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {result.prediction.toUpperCase()} CANDIDATE
                </h3>
              </div>
              <p className="text-muted-foreground">
                Confidence: <span className="text-lg font-bold text-foreground">{(result.confidence * 100).toFixed(1)}%</span>
              </p>
              <div className="w-full max-w-sm space-y-1.5">
                <div className="flex justify-between text-xs sm:text-sm">
                  <span className="text-muted-foreground">HER2 Inhibition Probability</span>
                  <span className="font-semibold text-primary">{(result.probability * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.probability * 100} className="h-2.5" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <FlaskConical className="h-4 w-4 text-primary" aria-hidden="true" />
              Drug-Likeness Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <LipinskiRadarChart lipinski={result.lipinski} />
          </CardContent>
        </Card>
        {result.descriptors && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-primary" aria-hidden="true" />
                Molecular Properties
              </CardTitle>
            </CardHeader>
            <CardContent>
              <DescriptorsBarChart descriptors={result.descriptors} />
            </CardContent>
          </Card>
        )}
      </div>

      {/* Descriptors Table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Beaker className="h-4 w-4 text-primary" aria-hidden="true" />
            Detailed Molecular Descriptors
          </CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Descriptor</TableHead>
                <TableHead className="text-right">Value</TableHead>
                <TableHead className="text-right">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">Molecular Weight</TableCell>
                <TableCell className="text-right tabular-nums">{result.lipinski.MW} Da</TableCell>
                <TableCell className="text-right">
                  <PropertyBadge status={getPropertyStatus(result.lipinski.MW, 500)}>
                    {result.lipinski.MW <= 500 ? "Optimal" : "High"}
                  </PropertyBadge>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">LogP (Lipophilicity)</TableCell>
                <TableCell className="text-right tabular-nums">{result.lipinski.LogP}</TableCell>
                <TableCell className="text-right">
                  <PropertyBadge status={getPropertyStatus(result.lipinski.LogP, 5)}>
                    {result.lipinski.LogP <= 5 ? "Optimal" : "High"}
                  </PropertyBadge>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">H-Bond Donors</TableCell>
                <TableCell className="text-right tabular-nums">{result.lipinski.NumHDonors}</TableCell>
                <TableCell className="text-right">
                  <PropertyBadge status={getPropertyStatus(result.lipinski.NumHDonors, 5)}>
                    {result.lipinski.NumHDonors <= 5 ? "Optimal" : "High"}
                  </PropertyBadge>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">H-Bond Acceptors</TableCell>
                <TableCell className="text-right tabular-nums">{result.lipinski.NumHAcceptors}</TableCell>
                <TableCell className="text-right">
                  <PropertyBadge status={getPropertyStatus(result.lipinski.NumHAcceptors, 10)}>
                    {result.lipinski.NumHAcceptors <= 10 ? "Optimal" : "High"}
                  </PropertyBadge>
                </TableCell>
              </TableRow>
              {result.descriptors && (
                <>
                  <TableRow>
                    <TableCell className="font-medium">TPSA (Polar Surface Area)</TableCell>
                    <TableCell className="text-right tabular-nums">{result.descriptors.TPSA} Ų</TableCell>
                    <TableCell className="text-right">
                      <PropertyBadge status={getPropertyStatus(result.descriptors.TPSA, 140)}>
                        {result.descriptors.TPSA <= 140 ? "Optimal" : "High"}
                      </PropertyBadge>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Rotatable Bonds</TableCell>
                    <TableCell className="text-right tabular-nums">{result.descriptors.NumRotatableBonds}</TableCell>
                    <TableCell className="text-right">
                      <PropertyBadge status={getPropertyStatus(result.descriptors.NumRotatableBonds, 10)}>
                        {result.descriptors.NumRotatableBonds <= 10 ? "Optimal" : "High"}
                      </PropertyBadge>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Aromatic Rings</TableCell>
                    <TableCell className="text-right tabular-nums">{result.descriptors.NumAromaticRings}</TableCell>
                    <TableCell className="text-right">
                      <PropertyBadge status={getPropertyStatus(result.descriptors.NumAromaticRings, 3)}>
                        {result.descriptors.NumAromaticRings <= 3 ? "Optimal" : "High"}
                      </PropertyBadge>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Heavy Atoms</TableCell>
                    <TableCell className="text-right tabular-nums">{result.descriptors.NumHeavyAtoms}</TableCell>
                    <TableCell className="text-right">
                      <PropertyBadge status="info" showIcon={false}>Info</PropertyBadge>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Molecular Formula</TableCell>
                    <TableCell className="text-right font-mono text-sm">{result.descriptors.MolecularFormula}</TableCell>
                    <TableCell className="text-right">
                      <PropertyBadge status="info" showIcon={false}>Info</PropertyBadge>
                    </TableCell>
                  </TableRow>
                </>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
