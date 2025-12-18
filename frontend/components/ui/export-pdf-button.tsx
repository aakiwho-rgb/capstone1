"use client";

import { useState } from "react";
import { PredictionResponse } from "@/types/api";
import { api } from "@/lib/api";
import { Button } from "./button";
import { FileDown, Loader2 } from "lucide-react";

interface ExportPDFButtonProps {
    prediction: PredictionResponse;
}

export function ExportPDFButton({ prediction }: ExportPDFButtonProps) {
    const [exporting, setExporting] = useState(false);

    const handleExport = async () => {
        try {
            setExporting(true);
            const response = await api.exportPDF(prediction);

            // Decode base64 and create blob
            const binaryString = atob(response.pdf);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const blob = new Blob([bytes], { type: response.mime_type });

            // Create download link
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = response.filename;
            document.body.appendChild(link);
            link.click();

            // Cleanup
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (err: any) {
            alert(err.message || "Failed to export PDF");
        } finally {
            setExporting(false);
        }
    };

    return (
        <Button
            onClick={handleExport}
            disabled={exporting}
            variant="outline"
            size="sm"
        >
            {exporting ? (
                <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                </>
            ) : (
                <>
                    <FileDown className="mr-2 h-4 w-4" />
                    Export PDF
                </>
            )}
        </Button>
    );
}
