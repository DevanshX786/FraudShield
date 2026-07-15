import React from "react";
import { Info, HelpCircle } from "lucide-react";

export default function ShapExplainer({ shapExplanation }) {
  if (!shapExplanation || Object.keys(shapExplanation).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-6 text-center text-muted-foreground bg-muted/20 border border-dashed rounded-lg border-border/80 min-h-[300px]">
        <Info className="w-8 h-8 mb-2 text-muted-foreground/50" />
        <span className="text-[13px] font-medium">No Explanations Available</span>
        <p className="text-[11px] mt-1 max-w-[200px] leading-relaxed">
          Select a transaction from the live feed to inspect its SHAP feature attribution.
        </p>
      </div>
    );
  }

  // Parse explanation and sort by absolute contribution
  const features = Object.entries(shapExplanation)
    .map(([name, value]) => ({
      name,
      value: parseFloat(value),
      absVal: Math.abs(parseFloat(value)),
    }))
    .sort((a, b) => b.absVal - a.absVal);

  const maxAbsVal = Math.max(...features.map((f) => f.absVal), 0.01);

  return (
    <div className="flex flex-col h-full bg-card border border-border/80 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-border/60">
        <div>
          <h3 className="text-[14px] font-bold text-foreground">
            SHAP Attribution Explainer
          </h3>
          <span className="text-[10px] text-muted-foreground">
            Local feature contributions for this decision
          </span>
        </div>
        <div className="group relative cursor-pointer">
          <HelpCircle className="w-4 h-4 text-muted-foreground/60 hover:text-foreground transition-colors" />
          <div className="absolute right-0 top-6 hidden group-hover:block bg-popover text-popover-foreground border border-border p-2.5 rounded shadow-lg text-[10px] w-56 z-50 leading-relaxed">
            SHAP (SHapley Additive exPlanations) shows how much each feature contributed to the model's final risk score compared to the average.
          </div>
        </div>
      </div>

      {/* Explanation Guide */}
      <div className="grid grid-cols-2 gap-4 text-center text-[10px] mb-4 bg-muted/30 p-2.5 rounded-lg border border-border/40">
        <div className="flex items-center justify-center gap-1.5 border-r border-border/50 text-emerald-600 dark:text-emerald-400 font-semibold">
          <div className="w-2.5 h-2.5 rounded bg-emerald-500/25 border border-emerald-500/30" />
          <span>Negative (Clean Indicator)</span>
        </div>
        <div className="flex items-center justify-center gap-1.5 text-rose-600 dark:text-rose-400 font-semibold">
          <div className="w-2.5 h-2.5 rounded bg-rose-500/25 border border-rose-500/30" />
          <span>Positive (Fraud Indicator)</span>
        </div>
      </div>

      {/* Feature Contributions List */}
      <div className="flex-1 flex flex-col gap-3 overflow-y-auto max-h-[400px] pr-1">
        {features.map((f) => {
          const isPositive = f.value > 0;
          const percentage = (f.absVal / maxAbsVal) * 100;
          
          return (
            <div key={f.name} className="flex flex-col text-left">
              <div className="flex items-center justify-between text-[11px] mb-1 font-medium text-foreground/80">
                <span className="font-mono truncate max-w-[170px]" title={f.name}>
                  {f.name}
                </span>
                <span className={`font-mono font-bold ${isPositive ? "text-rose-500" : "text-emerald-500"}`}>
                  {isPositive ? `+${f.value.toFixed(4)}` : f.value.toFixed(4)}
                </span>
              </div>

              {/* Custom Bidirectional Bar Chart */}
              <div className="h-3 w-full bg-muted/60 dark:bg-muted/30 rounded-sm relative overflow-hidden flex">
                {/* Left Side (Negative/Clean) */}
                <div className="w-1/2 flex justify-end h-full relative">
                  {!isPositive && (
                    <div
                      className="h-full bg-emerald-500/80 rounded-l-sm transition-all duration-300"
                      style={{ width: `${percentage}%` }}
                    />
                  )}
                </div>
                {/* Right Side (Positive/Fraud) */}
                <div className="w-1/2 flex justify-start h-full relative border-l border-border/80">
                  {isPositive && (
                    <div
                      className="h-full bg-rose-500/80 rounded-r-sm transition-all duration-300"
                      style={{ width: `${percentage}%` }}
                    />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
