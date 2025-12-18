"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const statusBadgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        success: "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20 dark:text-emerald-400 dark:bg-emerald-500/20",
        warning: "bg-amber-500/10 text-amber-600 border border-amber-500/20 dark:text-amber-400 dark:bg-amber-500/20",
        error: "bg-red-500/10 text-red-600 border border-red-500/20 dark:text-red-400 dark:bg-red-500/20",
        info: "bg-blue-500/10 text-blue-600 border border-blue-500/20 dark:text-blue-400 dark:bg-blue-500/20",
        neutral: "bg-muted text-muted-foreground border border-border",
      },
    },
    defaultVariants: {
      variant: "neutral",
    },
  }
);

export interface StatusBadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof statusBadgeVariants> {
  pulse?: boolean;
}

/**
 * StatusBadge - A semantic badge for displaying status indicators
 * Supports success, warning, error, info, and neutral variants with optional pulse animation
 */
export function StatusBadge({
  className,
  variant,
  pulse = false,
  children,
  ...props
}: StatusBadgeProps) {
  return (
    <span className={cn(statusBadgeVariants({ variant }), className)} {...props}>
      {pulse && (
        <span
          className={cn(
            "h-2 w-2 rounded-full animate-pulse",
            variant === "success" && "bg-emerald-500",
            variant === "warning" && "bg-amber-500",
            variant === "error" && "bg-red-500",
            variant === "info" && "bg-blue-500",
            variant === "neutral" && "bg-muted-foreground"
          )}
          aria-hidden="true"
        />
      )}
      {children}
    </span>
  );
}
