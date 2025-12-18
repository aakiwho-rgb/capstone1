"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Check, AlertTriangle, Info } from "lucide-react";

type PropertyStatus = "optimal" | "warning" | "info";

interface PropertyBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  status: PropertyStatus;
  showIcon?: boolean;
}

const statusConfig = {
  optimal: {
    icon: Check,
    label: "Optimal",
    className: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400",
  },
  warning: {
    icon: AlertTriangle,
    label: "Warning",
    className: "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:text-amber-400",
  },
  info: {
    icon: Info,
    label: "Info",
    className: "bg-muted text-muted-foreground border-border",
  },
};

/**
 * PropertyBadge - Displays status indicators for molecular properties
 * Shows optimal/warning/info states with consistent styling
 */
export function PropertyBadge({
  status,
  showIcon = true,
  children,
  className,
  ...props
}: PropertyBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
        config.className,
        className
      )}
      {...props}
    >
      {showIcon && <Icon className="h-3 w-3" aria-hidden="true" />}
      {children || config.label}
    </span>
  );
}
