"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

interface LoadingStateProps {
  variant?: "card" | "inline" | "full";
  className?: string;
  lines?: number;
}

/**
 * LoadingState - Consistent loading skeletons for different UI contexts
 * Variants: card (for card content), inline (for inline elements), full (for full sections)
 */
export function LoadingState({
  variant = "card",
  className,
  lines = 4,
}: LoadingStateProps) {
  if (variant === "inline") {
    return (
      <div className={cn("space-y-2", className)}>
        {Array.from({ length: lines }).map((_, i) => (
          <Skeleton
            key={i}
            className={cn("h-4", i === lines - 1 ? "w-3/4" : "w-full")}
          />
        ))}
      </div>
    );
  }

  if (variant === "full") {
    return (
      <div className={cn("animate-pulse space-y-6", className)}>
        <div className="space-y-3">
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </div>
        <Skeleton className="h-[200px] w-full rounded-xl" />
        <div className="space-y-3">
          {Array.from({ length: lines }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-full" />
          ))}
        </div>
      </div>
    );
  }

  // Default card variant
  return (
    <Card className={cn("animate-pulse", className)}>
      <CardHeader className="space-y-2">
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </CardHeader>
      <CardContent className="space-y-3">
        <Skeleton className="h-[180px] w-full rounded-lg" />
        <div className="space-y-2">
          {Array.from({ length: lines }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-full" />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
