"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import type { LipinskiStats } from "@/types/api";
import { cn } from "@/lib/utils";

interface LipinskiRadarChartProps {
  lipinski: LipinskiStats;
  className?: string;
}

// Lipinski Rule of 5 thresholds
const LIPINSKI_THRESHOLDS = {
  MW: 500, // Molecular Weight <= 500 Da
  LogP: 5, // LogP <= 5
  NumHDonors: 5, // H-bond donors <= 5
  NumHAcceptors: 10, // H-bond acceptors <= 10
};

/**
 * LipinskiRadarChart - Visualizes Lipinski Rule of 5 compliance
 * Supports dark mode with semantic theming
 */
export function LipinskiRadarChart({
  lipinski,
  className = "",
}: LipinskiRadarChartProps) {
  // Normalize values to percentage of threshold (capped at 150% for visualization)
  const data = [
    {
      property: "MW",
      fullName: "Molecular Weight",
      value: Math.min((lipinski.MW / LIPINSKI_THRESHOLDS.MW) * 100, 150),
      actual: lipinski.MW,
      threshold: LIPINSKI_THRESHOLDS.MW,
      unit: "Da",
      compliant: lipinski.MW <= LIPINSKI_THRESHOLDS.MW,
    },
    {
      property: "LogP",
      fullName: "LogP (Lipophilicity)",
      value: Math.min(
        (Math.abs(lipinski.LogP) / LIPINSKI_THRESHOLDS.LogP) * 100,
        150
      ),
      actual: lipinski.LogP,
      threshold: LIPINSKI_THRESHOLDS.LogP,
      unit: "",
      compliant: lipinski.LogP <= LIPINSKI_THRESHOLDS.LogP,
    },
    {
      property: "HBD",
      fullName: "H-Bond Donors",
      value: Math.min(
        (lipinski.NumHDonors / LIPINSKI_THRESHOLDS.NumHDonors) * 100,
        150
      ),
      actual: lipinski.NumHDonors,
      threshold: LIPINSKI_THRESHOLDS.NumHDonors,
      unit: "",
      compliant: lipinski.NumHDonors <= LIPINSKI_THRESHOLDS.NumHDonors,
    },
    {
      property: "HBA",
      fullName: "H-Bond Acceptors",
      value: Math.min(
        (lipinski.NumHAcceptors / LIPINSKI_THRESHOLDS.NumHAcceptors) * 100,
        150
      ),
      actual: lipinski.NumHAcceptors,
      threshold: LIPINSKI_THRESHOLDS.NumHAcceptors,
      unit: "",
      compliant: lipinski.NumHAcceptors <= LIPINSKI_THRESHOLDS.NumHAcceptors,
    },
  ];

  const violations = data.filter((d) => !d.compliant).length;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="rounded-lg border border-border bg-popover p-3 shadow-lg text-sm">
          <p className="font-semibold text-popover-foreground">{item.fullName}</p>
          <p className="text-muted-foreground">
            Value:{" "}
            <span className="font-medium text-popover-foreground">
              {item.actual}
              {item.unit}
            </span>
          </p>
          <p className="text-muted-foreground">
            Threshold:{" "}
            <span className="font-medium text-popover-foreground">
              ≤{item.threshold}
              {item.unit}
            </span>
          </p>
          <p className={item.compliant ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
            {item.compliant ? "✓ Compliant" : "✗ Violation"}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={cn("w-full", className)} role="img" aria-label="Lipinski Rule of 5 compliance chart">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-foreground">Lipinski Rule of 5</h4>
        <span
          className={cn(
            "text-xs font-medium px-2.5 py-1 rounded-full border",
            violations === 0
              ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400"
              : violations <= 1
              ? "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:text-amber-400"
              : "bg-red-500/10 text-red-600 border-red-500/20 dark:text-red-400"
          )}
        >
          {violations === 0
            ? "Drug-like"
            : `${violations} violation${violations > 1 ? "s" : ""}`}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={data} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
          <PolarGrid 
            stroke="hsl(var(--border))" 
            strokeOpacity={0.5}
          />
          <PolarAngleAxis
            dataKey="property"
            tick={{ 
              fontSize: 12, 
              fill: "hsl(var(--muted-foreground))",
              fontWeight: 500
            }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 150]}
            tick={{ 
              fontSize: 10, 
              fill: "hsl(var(--muted-foreground))" 
            }}
            tickFormatter={(value) => `${value}%`}
            axisLine={false}
          />
          {/* Threshold line at 100% */}
          <Radar
            name="Threshold"
            dataKey={() => 100}
            stroke="hsl(var(--muted-foreground))"
            fill="none"
            strokeWidth={2}
            strokeDasharray="5 5"
            strokeOpacity={0.6}
          />
          {/* Actual values */}
          <Radar
            name="Molecule"
            dataKey="value"
            stroke="hsl(var(--primary))"
            fill="hsl(var(--primary))"
            fillOpacity={0.25}
            strokeWidth={2}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }}
            formatter={(value) => (
              <span className="text-muted-foreground">{value}</span>
            )}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
