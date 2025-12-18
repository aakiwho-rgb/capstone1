"use client";

import { useState } from "react";
import {
  History,
  ChevronRight,
  ChevronLeft,
  CheckCircle2,
  XCircle,
  Trash2,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { PredictionResponse } from "@/types/api";

export interface HistoryEntry {
  id: string;
  smiles: string;
  prediction: "Active" | "Inactive";
  confidence: number;
  timestamp: Date;
  moleculeName?: string;
}

interface HistorySidebarProps {
  history: HistoryEntry[];
  onSelectEntry: (entry: HistoryEntry) => void;
  onClearHistory: () => void;
  className?: string;
}

/**
 * HistorySidebar - Collapsible sidebar showing recent prediction runs
 * Allows scientists to quickly compare previous analyses
 */
export function HistorySidebar({
  history,
  onSelectEntry,
  onClearHistory,
  className,
}: HistorySidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const formatTime = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    return date.toLocaleDateString();
  };

  const truncateSmiles = (smiles: string, maxLen = 20) => {
    if (smiles.length <= maxLen) return smiles;
    return smiles.slice(0, maxLen) + "...";
  };

  return (
    <div
      className={cn(
        "flex flex-col bg-card border-l border-border transition-all duration-300 ease-in-out",
        isCollapsed ? "w-12" : "w-72",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <History className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Recent Runs</span>
            <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded-full">
              {history.length}
            </span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto">
        {isCollapsed ? (
          // Collapsed view - just icons
          <div className="flex flex-col items-center gap-1 p-2">
            {history.slice(0, 10).map((entry) => (
              <button
                key={entry.id}
                onClick={() => onSelectEntry(entry)}
                className={cn(
                  "w-8 h-8 rounded-lg flex items-center justify-center transition-colors",
                  entry.prediction === "Active"
                    ? "bg-teal-500/10 hover:bg-teal-500/20 text-teal-500"
                    : "bg-rose-500/10 hover:bg-rose-500/20 text-rose-500"
                )}
                aria-label={`${entry.prediction} - ${entry.smiles}`}
              >
                {entry.prediction === "Active" ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  <XCircle className="h-4 w-4" />
                )}
              </button>
            ))}
          </div>
        ) : (
          // Expanded view - full details
          <div className="p-2 space-y-1">
            {history.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <Clock className="h-8 w-8 text-muted-foreground/30 mb-2" />
                <p className="text-xs text-muted-foreground">No predictions yet</p>
                <p className="text-[10px] text-muted-foreground/60 mt-1">
                  Run a prediction to see history
                </p>
              </div>
            ) : (
              history.map((entry) => (
                <button
                  key={entry.id}
                  onClick={() => onSelectEntry(entry)}
                  className={cn(
                    "w-full text-left p-2.5 rounded-lg border transition-all",
                    "hover:shadow-sm",
                    entry.prediction === "Active"
                      ? "border-teal-500/20 bg-teal-500/5 hover:bg-teal-500/10 hover:border-teal-500/30"
                      : "border-rose-500/20 bg-rose-500/5 hover:bg-rose-500/10 hover:border-rose-500/30"
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      {entry.moleculeName && (
                        <p className="text-xs font-medium text-foreground truncate">
                          {entry.moleculeName}
                        </p>
                      )}
                      <p className="text-[10px] font-mono text-muted-foreground truncate">
                        {truncateSmiles(entry.smiles, 25)}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      {entry.prediction === "Active" ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-teal-500" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 text-rose-500" />
                      )}
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-1.5">
                    <span
                      className={cn(
                        "text-[10px] font-semibold uppercase tracking-wider",
                        entry.prediction === "Active"
                          ? "text-teal-600 dark:text-teal-400"
                          : "text-rose-600 dark:text-rose-400"
                      )}
                    >
                      {entry.prediction}
                    </span>
                    <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                      <span>{(entry.confidence * 100).toFixed(0)}%</span>
                      <span>·</span>
                      <span>{formatTime(entry.timestamp)}</span>
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {/* Footer - Clear button */}
      {!isCollapsed && history.length > 0 && (
        <div className="p-2 border-t border-border">
          <Button
            variant="ghost"
            size="sm"
            className="w-full h-8 text-xs text-muted-foreground hover:text-destructive"
            onClick={onClearHistory}
          >
            <Trash2 className="h-3 w-3 mr-1.5" />
            Clear History
          </Button>
        </div>
      )}
    </div>
  );
}
