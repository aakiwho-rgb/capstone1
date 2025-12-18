"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, Atom } from "lucide-react";
import { cn } from "@/lib/utils";

interface MoleculeViewerProps {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
}

/**
 * MoleculeViewer - Renders 2D molecular structure from SMILES
 * Uses backend RDKit SVG generation for reliable cross-browser support
 */
export function MoleculeViewer({
  smiles,
  width = 300,
  height = 300,
  className = "",
}: MoleculeViewerProps) {
  const [svg, setSvg] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!smiles) {
      setLoading(false);
      setError("No SMILES provided");
      return;
    }

    const fetchSVG = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await api.getMoleculeSVG(smiles, width, height);
        setSvg(response.svg);
      } catch (err: any) {
        console.error("Failed to generate molecule SVG:", err);
        setError(err.message || "Failed to render molecule");
      } finally {
        setLoading(false);
      }
    };

    fetchSVG();
  }, [smiles, width, height]);

  if (loading) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-lg bg-muted/50",
          className
        )}
        style={{ width, height }}
        aria-label="Loading molecule structure"
        role="status"
      >
        <div className="flex flex-col items-center gap-2">
          <Atom className="h-8 w-8 text-muted-foreground/50 animate-pulse" />
          <span className="text-xs text-muted-foreground">Rendering...</span>
        </div>
      </div>
    );
  }

  if (error || !svg) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center bg-muted/50 rounded-lg text-muted-foreground",
          className
        )}
        style={{ width, height }}
        role="img"
        aria-label={`Unable to render molecule: ${error || "Unknown error"}`}
      >
        <AlertCircle className="h-8 w-8 mb-2 opacity-50" aria-hidden="true" />
        <span className="text-xs text-center px-4">
          {error || "Unable to render molecule"}
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "molecule-viewer bg-card rounded-lg border border-border overflow-hidden",
        className
      )}
      style={{ width, height }}
      role="img"
      aria-label={`2D structure of molecule with SMILES: ${smiles.substring(0, 50)}${smiles.length > 50 ? "..." : ""}`}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
