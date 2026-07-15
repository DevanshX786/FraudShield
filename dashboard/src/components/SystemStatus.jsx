import React, { useState } from "react";
import {
  Activity,
  Database,
  Cpu,
  RefreshCw,
  AlertTriangle,
  Play,
  ShieldCheck
} from "lucide-react";

export default function SystemStatus({ health, modelInfo, onTriggerRetrain, demoActive, demoStatus }) {
  const [retrainStatus, setRetrainStatus] = useState("idle");

  const handleRetrain = async () => {
    setRetrainStatus("loading");
    const success = await onTriggerRetrain();
    if (success) {
      setRetrainStatus("started");
      setTimeout(() => setRetrainStatus("idle"), 5000);
    } else {
      setRetrainStatus("failed");
      setTimeout(() => setRetrainStatus("idle"), 4000);
    }
  };

  const healthIndicators = [
    {
      name: "FastAPI Endpoint serving",
      status: health ? "Healthy" : "Offline",
      description: "Serves live transaction risk scores",
      icon: Activity,
      ok: !!health
    },
    {
      name: "PostgreSQL Database",
      status: health?.database === "connected" ? "Connected" : "Offline",
      description: "Stores features, logs, and registry entries",
      icon: Database,
      ok: health?.database === "connected"
    },
    {
      name: "XGBoost Classifier",
      status: health?.model === "loaded" ? "Loaded" : "Missing",
      description: "Active ML model serving inferences",
      icon: Cpu,
      ok: health?.model === "loaded"
    }
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Health Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {healthIndicators.map((card) => {
          const Icon = card.icon;
          return (
            <div
              key={card.name}
              className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left"
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                  {card.name}
                </span>
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    card.ok ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                  }`}
                />
              </div>

              <div className="flex items-center gap-3">
                <div
                  className={`p-2 rounded-lg border ${
                    card.ok
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                      : "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-foreground">
                    {card.status}
                  </span>
                  <span className="text-[10px] text-muted-foreground mt-0.5 leading-none">
                    {card.description}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Retraining Controls Panel */}
      <div className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left">
        <div className="mb-4">
          <h3 className="text-[14px] font-bold text-foreground">Retraining Control Panel</h3>
          <span className="text-[11px] text-muted-foreground font-medium">
            Trigger statistical model updates and promotion evaluations
          </span>
        </div>

        <div className="flex flex-col lg:flex-row gap-6 items-start lg:items-center justify-between border border-border/60 bg-muted/20 p-5 rounded-lg">
          <div className="flex items-start gap-4 max-w-xl text-left">
            <div className="p-2.5 rounded-lg border border-amber-500/25 bg-amber-500/10 text-amber-600 dark:text-amber-400 shrink-0">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-bold text-foreground">System Retraining Warning</span>
              <p className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
                Manually starting retraining invokes the full preprocessing, scaling, SMOTE
                oversampling, and XGBoost grid search optimization flow. This is an asynchronous
                task and may take 30-60 seconds depending on serving server load.
              </p>
            </div>
          </div>

          <button
            onClick={handleRetrain}
            disabled={retrainStatus === "loading" || !health}
            className={`flex items-center gap-1.5 px-4 py-2.5 rounded-md text-xs font-semibold shadow-xs select-none transition-all duration-150 shrink-0 ${
              retrainStatus === "loading"
                ? "bg-muted text-muted-foreground border border-border"
                : retrainStatus === "started"
                ? "bg-emerald-500 text-white hover:bg-emerald-600"
                : "bg-primary text-primary-foreground hover:bg-primary/95"
            }`}
          >
            {retrainStatus === "loading" ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Submitting request...</span>
              </>
            ) : retrainStatus === "started" ? (
              <>
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Retraining Started!</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>Trigger Retrain Pipeline</span>
              </>
            )}
          </button>
        </div>

        {retrainStatus === "started" && (
          <div className="mt-4 p-3 bg-emerald-500/15 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 rounded-lg text-xs leading-normal text-left">
            <span className="font-semibold">Workflow running in background.</span>
            <p className="text-[10px] mt-0.5 text-emerald-500/80">
              The model retraining run is running on the FastAPI backend. If the resulting F1 score
              improves the active version by &gt;1.0%, the model registry will promote the version automatically.
            </p>
          </div>
        )}

        {retrainStatus === "failed" && (
          <div className="mt-4 p-3 bg-rose-500/15 border border-rose-500/20 text-rose-600 dark:text-rose-400 rounded-lg text-xs leading-normal text-left">
            <span className="font-semibold">Pipeline request failed.</span>
            <p className="text-[10px] mt-0.5 text-rose-500/80">
              Ensure the serving API is online and the model retrain mutex locks are clear.
            </p>
          </div>
        )}
      </div>

      {/* Live Demo Logs Terminal */}
      {demoActive && demoStatus?.events && (
        <div className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-foreground">MLOps Live Activity Log</h3>
              <span className="text-[11px] text-muted-foreground">
                Real-time streaming pipeline event logs from DemoScenarioEngine
              </span>
            </div>
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
          </div>

          <div className="font-mono text-[11px] text-slate-300 bg-slate-950 dark:bg-slate-900 border border-slate-800 rounded-lg p-4 h-60 overflow-y-auto flex flex-col gap-1.5 shadow-inner">
            {demoStatus.events.length === 0 ? (
              <span className="text-slate-500 italic">Awaiting logs...</span>
            ) : (
              demoStatus.events.map((evt, idx) => {
                let colorClass = "text-slate-300";
                if (evt.includes("ALERT") || evt.includes("degraded") || evt.includes("drift_detected")) {
                  colorClass = "text-rose-400 font-semibold";
                } else if (evt.includes("passed") || evt.includes("deployed") || evt.includes("resolved") || evt.includes("Completed")) {
                  colorClass = "text-emerald-400 font-semibold";
                } else if (evt.includes("Testing") || evt.includes("Training") || evt.includes("Best")) {
                  colorClass = "text-indigo-400";
                }
                return (
                  <div key={idx} className={`${colorClass} leading-normal`}>
                    {evt}
                  </div>
                );
              })
            )}
            <div ref={(el) => el?.scrollIntoView({ behavior: "smooth" })} />
          </div>
        </div>
      )}

      {/* Embedded Grafana iframe or simulated panels */}
      <div className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left">
        <div className="mb-4">
          <h3 className="text-sm font-bold text-foreground">Grafana Cloud Service Health</h3>
          <span className="text-[11px] text-muted-foreground">
            Grafana metrics streaming live from FastAPI /grafana-metrics endpoint
          </span>
        </div>
        
        {/* Visual mock of Grafana Panel when iframe is offline */}
        <div className="border border-dashed border-border/80 bg-muted/10 rounded-lg p-8 text-center flex flex-col items-center justify-center min-h-[160px]">
          <Database className="w-8 h-8 text-muted-foreground/45 mb-2" />
          <p className="text-xs text-foreground font-semibold">Grafana Live Monitoring</p>
          <span className="text-[10px] text-muted-foreground max-w-sm mt-1 leading-normal">
            Local metrics are exported directly at <code className="bg-muted px-1 py-0.5 rounded font-mono font-bold">/grafana-metrics</code> for Grafana API ingest. If local Grafana is running, panels will automatically stream details.
          </span>
        </div>
      </div>
    </div>
  );
}
