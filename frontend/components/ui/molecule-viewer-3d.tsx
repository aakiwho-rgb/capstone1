"use client";

import { useEffect, useRef, useState } from "react";
import { Atom, RotateCcw, Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface MoleculeViewer3DProps {
  smiles: string;
  className?: string;
}

/**
 * MoleculeViewer3D - Placeholder for 3D molecular visualization
 * 
 * Integration notes:
 * - Use 3Dmol.js (https://3dmol.csb.pitt.edu/) for WebGL-based 3D rendering
 * - Alternative: Mol* (https://molstar.org/) for more advanced features
 * - Backend would need to convert SMILES → 3D coordinates (RDKit AllChem.EmbedMolecule)
 * - Or use NGL Viewer for PDB/SDF file support
 * 
 * Example 3Dmol.js integration:
 * ```
 * import $3Dmol from '3dmol';
 * const viewer = $3Dmol.createViewer(containerRef.current, { backgroundColor: 'transparent' });
 * viewer.addModel(sdfData, 'sdf');
 * viewer.setStyle({}, { stick: { colorscheme: 'cyanCarbon' } });
 * viewer.zoomTo();
 * viewer.render();
 * ```
 */
export function MoleculeViewer3D({ smiles, className }: MoleculeViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isRotating, setIsRotating] = useState(true);
  const [rotation, setRotation] = useState(0);

  // Simulated rotation animation for the placeholder
  useEffect(() => {
    if (!isRotating) return;
    const interval = setInterval(() => {
      setRotation((prev) => (prev + 1) % 360);
    }, 50);
    return () => clearInterval(interval);
  }, [isRotating]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative rounded-xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-slate-700/50 overflow-hidden",
        className
      )}
    >
      {/* Ambient glow effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-teal-500/10 via-transparent to-cyan-500/10 pointer-events-none" />
      
      {/* Grid overlay for scientific aesthetic */}
      <div 
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(148, 163, 184, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }}
      />

      {/* Placeholder 3D molecule representation */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div 
          className="relative"
          style={{ transform: `rotateY(${rotation}deg)`, transformStyle: 'preserve-3d' }}
        >
          {/* Central atom */}
          <div className="w-16 h-16 rounded-full bg-gradient-to-br from-teal-400 to-cyan-500 shadow-lg shadow-teal-500/30 flex items-center justify-center">
            <Atom className="w-8 h-8 text-white" />
          </div>
          
          {/* Orbital rings */}
          <div 
            className="absolute inset-0 -m-8 border-2 border-teal-400/30 rounded-full"
            style={{ transform: 'rotateX(60deg)' }}
          />
          <div 
            className="absolute inset-0 -m-12 border border-cyan-400/20 rounded-full"
            style={{ transform: 'rotateX(60deg) rotateZ(45deg)' }}
          />
          
          {/* Electron dots */}
          {[0, 72, 144, 216, 288].map((angle, i) => (
            <div
              key={i}
              className="absolute w-3 h-3 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"
              style={{
                left: '50%',
                top: '50%',
                transform: `translate(-50%, -50%) rotate(${angle + rotation * 2}deg) translateX(50px)`,
              }}
            />
          ))}
        </div>
      </div>

      {/* Control buttons */}
      <div className="absolute top-3 right-3 flex flex-col gap-1.5">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white backdrop-blur-sm"
          onClick={() => setIsRotating(!isRotating)}
          aria-label={isRotating ? "Stop rotation" : "Start rotation"}
        >
          <RotateCcw className={cn("h-3.5 w-3.5", isRotating && "animate-spin")} />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white backdrop-blur-sm"
          aria-label="Zoom in"
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white backdrop-blur-sm"
          aria-label="Zoom out"
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white backdrop-blur-sm"
          aria-label="Fullscreen"
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* SMILES label */}
      <div className="absolute bottom-3 left-3 right-3">
        <div className="bg-slate-900/90 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-700/50">
          <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-0.5">Structure</p>
          <p className="text-xs text-slate-200 font-mono truncate">{smiles}</p>
        </div>
      </div>

      {/* "3D" badge */}
      <div className="absolute top-3 left-3 bg-teal-500/20 border border-teal-500/30 rounded-md px-2 py-0.5">
        <span className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider">3D View</span>
      </div>
    </div>
  );
}
