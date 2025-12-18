"use client";

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
 * DescriptorsBarChart - Visualizes extended molecular descriptors
 * Supports dark mode with semantic theming
 */
export function DescriptorsBarChart({
  descriptors,
  className = "",
}: DescriptorsBarChartProps) {
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

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="rounded-lg border border-border bg-popover p-3 shadow-lg text-sm">
          <p className="font-semibold text-popover-foreground">{item.name}</p>
          <p className="text-muted-foreground">
            Value:{" "}
            <span className="font-medium text-popover-foreground">
              {item.value}
              {item.unit}
            </span>
          </p>
          <p
            className={
              item.isOptimal
                ? "text-emerald-600 dark:text-emerald-400"
                : "text-amber-600 dark:text-amber-400"
            }
          >
            {item.isOptimal ? "✓ Within optimal range" : "⚠ Outside optimal range"}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={cn("w-full", className)} role="img" aria-label="Molecular descriptors chart">
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
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 20, left: 70, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--border))"
            horizontal={false}
            strokeOpacity={0.5}
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            tickFormatter={(value) => `${value}%`}
            axisLine={{ stroke: "hsl(var(--border))" }}
            tickLine={{ stroke: "hsl(var(--border))" }}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))", fontWeight: 500 }}
            width={65}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(var(--muted))", opacity: 0.3 }} />
          <Bar dataKey="normalizedValue" radius={[0, 4, 4, 0]} maxBarSize={18}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.isOptimal ? "#10b981" : "#f59e0b"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
