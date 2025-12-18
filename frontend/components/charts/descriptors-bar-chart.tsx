"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ExtendedDescriptors } from "@/types/api";
import { cn } from "@/lib/utils";

interface DescriptorsBarChartProps {
  descriptors: ExtendedDescriptors;
  className?: string;
  compact?: boolean;
}

// Drug-likeness reference ranges (approximate optimal ranges)
const DESCRIPTOR_CONFIG = {
  MW: { label: "MW", unit: "Da", optimal: 500, max: 800 },
  TPSA: { label: "TPSA", unit: "Ų", optimal: 140, max: 200 },
  NumRotatableBonds: { label: "Rot. Bonds", unit: "", optimal: 10, max: 15 },
  NumHeavyAtoms: { label: "Heavy Atoms", unit: "", optimal: 35, max: 50 },
  NumAromaticRings: { label: "Aromatic Rings", unit: "", optimal: 3, max: 5 },
  FractionCSP3: { label: "Fsp³", unit: "", optimal: 0.25, max: 1.0 },
};

/**
 * Custom Tooltip component with theme support
 */
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  
  const data = payload[0].payload;
  
  return (
    <div className="bg-popover border border-border rounded-lg shadow-lg p-2 text-popover-foreground">
      <p className="font-medium text-sm">{data.name}</p>
      <p className="text-xs text-muted-foreground mt-1">
        Value: <span className="font-mono font-medium text-foreground">{data.value}{data.unit ? ` ${data.unit}` : ''}</span>
      </p>
      <p className={cn(
        "text-xs mt-0.5",
        data.isOptimal ? "text-teal-500" : "text-amber-500"
      )}>
        {data.isOptimal ? "✓ Within optimal range" : "⚠ Above optimal"}
      </p>
    </div>
  );
}

/**
 * DescriptorsBarChart - Visualizes extended molecular descriptors
 * Supports dark mode with semantic theming
 */
export function DescriptorsBarChart({
  descriptors,
  className = "",
  compact = false,
}: DescriptorsBarChartProps) {
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
    optimal: isDark ? "#2dd4bf" : "#14b8a6",  // teal
    warning: isDark ? "#fbbf24" : "#f59e0b",  // amber
    cursor: isDark ? "rgba(55, 65, 81, 0.5)" : "rgba(229, 231, 235, 0.5)",
  };

  const data = [
    {
      name: DESCRIPTOR_CONFIG.MW.label,
      value: descriptors.MW,
      normalizedValue: Math.min(
        (descriptors.MW / DESCRIPTOR_CONFIG.MW.max) * 100,
        100
      ),
      optimal: (DESCRIPTOR_CONFIG.MW.optimal / DESCRIPTOR_CONFIG.MW.max) * 100,
      unit: DESCRIPTOR_CONFIG.MW.unit,
      isOptimal: descriptors.MW <= DESCRIPTOR_CONFIG.MW.optimal,
    },
    {
      name: DESCRIPTOR_CONFIG.TPSA.label,
      value: descriptors.TPSA,
      normalizedValue: Math.min(
        (descriptors.TPSA / DESCRIPTOR_CONFIG.TPSA.max) * 100,
        100
      ),
      optimal:
        (DESCRIPTOR_CONFIG.TPSA.optimal / DESCRIPTOR_CONFIG.TPSA.max) * 100,
      unit: DESCRIPTOR_CONFIG.TPSA.unit,
      isOptimal: descriptors.TPSA <= DESCRIPTOR_CONFIG.TPSA.optimal,
    },
    {
      name: DESCRIPTOR_CONFIG.NumRotatableBonds.label,
      value: descriptors.NumRotatableBonds,
      normalizedValue: Math.min(
        (descriptors.NumRotatableBonds /
          DESCRIPTOR_CONFIG.NumRotatableBonds.max) *
          100,
        100
      ),
      optimal:
        (DESCRIPTOR_CONFIG.NumRotatableBonds.optimal /
          DESCRIPTOR_CONFIG.NumRotatableBonds.max) *
        100,
      unit: DESCRIPTOR_CONFIG.NumRotatableBonds.unit,
      isOptimal:
        descriptors.NumRotatableBonds <=
        DESCRIPTOR_CONFIG.NumRotatableBonds.optimal,
    },
    {
      name: DESCRIPTOR_CONFIG.NumHeavyAtoms.label,
      value: descriptors.NumHeavyAtoms,
      normalizedValue: Math.min(
        (descriptors.NumHeavyAtoms / DESCRIPTOR_CONFIG.NumHeavyAtoms.max) * 100,
        100
      ),
      optimal:
        (DESCRIPTOR_CONFIG.NumHeavyAtoms.optimal /
          DESCRIPTOR_CONFIG.NumHeavyAtoms.max) *
        100,
      unit: DESCRIPTOR_CONFIG.NumHeavyAtoms.unit,
      isOptimal:
        descriptors.NumHeavyAtoms <= DESCRIPTOR_CONFIG.NumHeavyAtoms.optimal,
    },
    {
      name: DESCRIPTOR_CONFIG.NumAromaticRings.label,
      value: descriptors.NumAromaticRings,
      normalizedValue: Math.min(
        (descriptors.NumAromaticRings / DESCRIPTOR_CONFIG.NumAromaticRings.max) *
          100,
        100
      ),
      optimal:
        (DESCRIPTOR_CONFIG.NumAromaticRings.optimal /
          DESCRIPTOR_CONFIG.NumAromaticRings.max) *
        100,
      unit: DESCRIPTOR_CONFIG.NumAromaticRings.unit,
      isOptimal:
        descriptors.NumAromaticRings <=
        DESCRIPTOR_CONFIG.NumAromaticRings.optimal,
    },
    {
      name: DESCRIPTOR_CONFIG.FractionCSP3.label,
      value: descriptors.FractionCSP3,
      normalizedValue: descriptors.FractionCSP3 * 100,
      optimal: DESCRIPTOR_CONFIG.FractionCSP3.optimal * 100,
      unit: DESCRIPTOR_CONFIG.FractionCSP3.unit,
      isOptimal: descriptors.FractionCSP3 >= DESCRIPTOR_CONFIG.FractionCSP3.optimal, // Higher is better for Fsp3
    },
  ];

  return (
    <div className={cn("w-full", className)} role="img" aria-label="Molecular descriptors chart">
      {!compact && (
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-foreground">
            Molecular Descriptors
          </h4>
          <span className="text-xs text-muted-foreground">
            Formula:{" "}
            <span className="font-mono font-medium text-foreground">
              {descriptors.MolecularFormula}
            </span>
          </span>
        </div>
      )}
      <ResponsiveContainer width="100%" height={compact ? 100 : 220}>
        <BarChart
          data={compact ? data.slice(0, 4) : data}
          layout="vertical"
          margin={compact ? { top: 0, right: 10, left: 50, bottom: 0 } : { top: 5, right: 20, left: 70, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={colors.grid}
            horizontal={false}
            strokeOpacity={0.5}
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fontSize: compact ? 8 : 10, fill: colors.text }}
            tickFormatter={(value) => `${value}%`}
            axisLine={{ stroke: colors.grid }}
            tickLine={{ stroke: colors.grid }}
            hide={compact}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: compact ? 9 : 11, fill: colors.text, fontWeight: 500 }}
            width={compact ? 45 : 65}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            content={<CustomTooltip />}
            cursor={{ fill: colors.cursor }}
          />
          <Bar dataKey="normalizedValue" radius={[0, 4, 4, 0]} maxBarSize={compact ? 14 : 18}>
            {(compact ? data.slice(0, 4) : data).map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.isOptimal ? colors.optimal : colors.warning}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
