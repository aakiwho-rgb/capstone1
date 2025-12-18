"use client";

import { useEffect, useState } from "react";
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
  compact?: boolean;
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
  compact = false,
}: LipinskiRadarChartProps) {
  // Theme detection for Recharts (which doesn't support CSS variables well)
  const [isDark, setIsDark] = useState(false);
  
  useEffect(() => {
    // Check initial theme
    const checkTheme = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    checkTheme();
    
    // Watch for theme changes
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { 
      attributes: true, 
      attributeFilter: ['class'] 
    });
    
    return () => observer.disconnect();
  }, []);

  // Theme-aware colors for Recharts
  const colors = {
    grid: isDark ? "#374151" : "#e5e7eb",
    text: isDark ? "#9ca3af" : "#6b7280",
    primary: isDark ? "#2dd4bf" : "#14b8a6",
    threshold: isDark ? "#6b7280" : "#9ca3af",
  };

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

  return (
    <div className={cn("w-full", className)} role="img" aria-label="Lipinski Rule of 5 compliance chart">
      {!compact && (
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
      )}
      <ResponsiveContainer width="100%" height={compact ? 100 : 260}>
        <RadarChart data={data} margin={compact ? { top: 5, right: 20, bottom: 5, left: 20 } : { top: 20, right: 30, bottom: 20, left: 30 }}>
          <PolarGrid 
            stroke={colors.grid} 
            strokeOpacity={0.7}
          />
          <PolarAngleAxis
            dataKey="property"
            tick={{ 
              fontSize: compact ? 9 : 12, 
              fill: colors.text,
              fontWeight: 500
            }}
          />
          {!compact && (
            <PolarRadiusAxis
              angle={90}
              domain={[0, 150]}
              tick={{ 
                fontSize: 10, 
                fill: colors.text 
              }}
              tickFormatter={(value) => `${value}%`}
              axisLine={false}
            />
          )}
          {/* Threshold line at 100% */}
          <Radar
            name="Threshold"
            dataKey={() => 100}
            stroke={colors.threshold}
            fill="none"
            strokeWidth={compact ? 1 : 2}
            strokeDasharray="5 5"
            strokeOpacity={0.8}
          />
          {/* Actual values */}
          <Radar
            name="Molecule"
            dataKey="value"
            stroke={colors.primary}
            fill={colors.primary}
            fillOpacity={0.3}
            strokeWidth={compact ? 1.5 : 2}
          />
          <Tooltip />
          {!compact && (
            <Legend
              wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }}
              formatter={(value) => (
                <span style={{ color: colors.text }}>{value}</span>
              )}
            />
          )}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
