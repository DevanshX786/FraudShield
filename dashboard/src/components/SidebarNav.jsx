import React, { useState } from "react";
import {
  LayoutDashboard,
  Activity,
  GitBranch,
  Terminal,
  ChevronDown,
  Sun,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  ShieldCheck
} from "lucide-react";

export function SidebarNav({
  activeView,
  onViewChange,
  isSidebarOpen,
  setSidebarOpen,
  theme,
  toggleTheme,
  systemStatus,
  modelInfo,
  demoActive,
  demoStatus,
  onRunDemo,
  onStopDemo,
  onSetDemoStage,
  onSetCustomMetrics,
  autoRetrain,
  setAutoRetrain
}) {
  const [selectedWorkspace, setSelectedWorkspace] = useState("FraudShield Main");
  const [isWorkspaceOpen, setIsWorkspaceOpen] = useState(false);
  const [isDemoExpanded, setIsDemoExpanded] = useState(false);
  const [localDrift, setLocalDrift] = useState(0.1);

  // Sync localDrift with external drift_score changes when not dragging
  React.useEffect(() => {
    if (demoStatus?.drift_score !== undefined) {
      setLocalDrift(demoStatus.drift_score);
    }
  }, [demoStatus?.drift_score]);

  const navGroups = [
    {
      heading: "Monitoring",
      items: [
        {
          id: "feed",
          title: "Live Feed",
          icon: LayoutDashboard,
          badge: "Stream"
        }
      ]
    },
    {
      heading: "MLOps Analytics",
      items: [
        {
          id: "metrics",
          title: "Model Performance",
          icon: Activity
        },
        {
          id: "drift",
          title: "Data Drift Monitor",
          icon: GitBranch,
          badge: systemStatus?.drift_detected ? "Alert" : null
        }
      ]
    },
    {
      heading: "Operations",
      items: [
        {
          id: "system",
          title: "System Control",
          icon: Terminal
        }
      ]
    }
  ];

  return (
    <div
      className={`flex flex-col h-full bg-card select-none transition-all duration-300 ease-in-out relative ${
        isSidebarOpen
          ? "w-[260px] p-3 border-r border-border/80 opacity-100"
          : "w-0 p-0 border-r-0 overflow-hidden opacity-0 pointer-events-none"
      }`}
    >
      {/* Workspace Switcher */}
      <div className="relative">
        <div
          onClick={() => setIsWorkspaceOpen(!isWorkspaceOpen)}
          className="flex items-center justify-between px-2.5 py-2 mb-4 rounded-lg hover:bg-muted/80 cursor-pointer transition-colors group"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm shadow-sm">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div className="flex flex-col overflow-hidden text-left">
              <span className="text-[13px] font-semibold leading-none mb-1 truncate text-foreground">
                {selectedWorkspace}
              </span>
              <span className="text-[10px] text-muted-foreground leading-none">
                Active System
              </span>
            </div>
          </div>
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground/50 group-hover:text-foreground/75 transition-colors" />
        </div>

        {isWorkspaceOpen && (
          <>
            <div
              className="fixed inset-0 z-40"
              onClick={() => setIsWorkspaceOpen(false)}
            />
            <div className="absolute top-[48px] left-0 w-full bg-popover border border-border rounded-lg shadow-lg z-50 py-1 flex flex-col gap-0.5 animate-in fade-in zoom-in-95 duration-100">
              {["FraudShield Main", "Staging Pipeline", "Audit Sandbox"].map(
                (ws) => (
                  <div
                    key={ws}
                    onClick={() => {
                      setSelectedWorkspace(ws);
                      setIsWorkspaceOpen(false);
                    }}
                    className={`px-3 py-1.5 mx-1 text-xs rounded-md cursor-pointer transition-colors text-left ${
                      selectedWorkspace === ws
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-foreground hover:bg-muted"
                    }`}
                  >
                    {ws}
                  </div>
                )
              )}
            </div>
          </>
        )}
      </div>

      {/* Navigation List */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-4 mt-2 pr-1">
        {navGroups.map((group, idx) => (
          <div key={idx} className="flex flex-col gap-0.5">
            <span className="px-2.5 mb-1 text-[10px] font-bold tracking-wider text-muted-foreground/50 uppercase text-left">
              {group.heading}
            </span>
            {group.items.map((item) => {
              const isActive = activeView === item.id;
              const Icon = item.icon;
              return (
                <div
                  key={item.id}
                  onClick={() => onViewChange(item.id)}
                  className={`group flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer transition-all duration-150 ${
                    isActive
                      ? "bg-primary text-primary-foreground font-medium shadow-sm"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className="w-4 h-4 shrink-0" strokeWidth={2} />
                    <span className="text-[13px] tracking-wide truncate">
                      {item.title}
                    </span>
                  </div>
                  {item.badge && (
                    <span
                      className={`flex items-center justify-center h-4.5 px-1.5 text-[9px] font-semibold rounded-full ${
                        item.badge === "Alert"
                          ? "bg-destructive text-destructive-foreground animate-pulse"
                          : "bg-emerald-500/10 text-emerald-500 dark:text-emerald-400"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Demo Controls Section */}
      <div className="border-t border-border/80 pt-3 mt-3 flex flex-col gap-1">
        <button
          onClick={() => setIsDemoExpanded(!isDemoExpanded)}
          className={`flex items-center justify-between px-2.5 py-2 rounded-md transition-colors cursor-pointer text-left ${
            demoActive
              ? "bg-indigo-500/10 text-indigo-400 font-semibold"
              : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
        >
          <div className="flex items-center gap-2.5">
            <Terminal className="w-4 h-4 shrink-0" strokeWidth={2} />
            <span className="text-[13px] tracking-wide font-medium">Demo Control System</span>
          </div>
          <ChevronDown
            className={`w-3.5 h-3.5 text-muted-foreground/50 transition-transform duration-200 ${
              isDemoExpanded ? "rotate-180" : ""
            }`}
          />
        </button>

        {isDemoExpanded && (
          <div className="flex flex-col gap-3 px-2.5 py-2.5 bg-muted/20 border border-border/60 rounded-lg mt-1 text-left animate-in slide-in-from-top-1 duration-150">
            {/* Master Demo Switch */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-muted-foreground">Demo Active State</span>
              <button
                onClick={demoActive ? onStopDemo : onRunDemo}
                className={`px-2 py-1 rounded text-[10px] font-bold cursor-pointer transition-colors ${
                  demoActive
                    ? "bg-rose-500/10 text-rose-500 hover:bg-rose-500/20 border border-rose-500/20"
                    : "bg-indigo-600 text-white hover:bg-indigo-500"
                }`}
              >
                {demoActive ? "STOP DEMO" : "START DEMO"}
              </button>
            </div>

            {demoActive && (
              <>
                <div className="h-px bg-border/40 my-0.5" />

                {/* Stage Select pills */}
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
                    Select Stage
                  </span>
                  <div className="grid grid-cols-5 gap-1">
                    {[
                      { stage: 1, label: "H", tooltip: "Healthy" },
                      { stage: 2, label: "D", tooltip: "Drift" },
                      { stage: 3, label: "A", tooltip: "Alert" },
                      { stage: 4, label: "R", tooltip: "Retrain" },
                      { stage: 5, label: "C", tooltip: "Recover" }
                    ].map((s) => (
                      <button
                        key={s.stage}
                        onClick={() => onSetDemoStage(s.stage)}
                        title={s.tooltip}
                        className={`py-1 rounded text-[10px] font-bold transition-all cursor-pointer border ${
                          demoStatus?.stage === s.stage
                            ? "bg-indigo-600 text-white border-indigo-600 shadow-xs"
                            : "bg-background text-muted-foreground border-border hover:bg-muted hover:text-foreground"
                        }`}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="h-px bg-border/40 my-0.5" />

                {/* Custom sliders */}
                <div className="flex flex-col gap-2.5">
                  <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
                    Drift Simulation Controls
                  </span>

                  {/* Drift Score Slider */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground">
                      <span>Drift Intensity</span>
                      <span className="font-bold text-indigo-400">
                        {localDrift.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={localDrift}
                      onChange={(e) => setLocalDrift(parseFloat(e.target.value))}
                      className="w-full h-1 bg-muted rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                  </div>

                  {/* Apply Drift button */}
                  <button
                    onClick={() => onSetCustomMetrics({ drift_score: localDrift })}
                    className="w-full py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-[10px] cursor-pointer shadow-xs transition-all"
                  >
                    Apply Drift Intensity
                  </button>

                  <div className="h-px bg-border/40 my-0.5" />

                  {/* Auto-retrain toggle */}
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px] font-semibold text-muted-foreground">Auto-Retrain on Alert</span>
                    <label className="relative inline-flex items-center cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={autoRetrain}
                        onChange={(e) => setAutoRetrain(e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-7 h-4 bg-muted peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-indigo-600"></div>
                    </label>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Bottom Panel controls / Theme Toggle */}
      <div className="mt-auto pt-4 border-t border-border/80 flex flex-col gap-2">
        {/* Retraining Alert */}
        {demoActive && demoStatus?.is_retraining && (
          <div className="flex items-center gap-2 px-3 py-2 bg-amber-500/15 border border-amber-500/25 rounded-lg text-amber-600 dark:text-amber-400 text-xs animate-pulse select-none text-left">
            <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0" />
            <span className="font-semibold text-[11px] leading-tight text-left">
              Model retraining in progress...
            </span>
          </div>
        )}

        {/* Deployment Alert */}
        {demoActive && modelInfo?.model_version === "v2" && (
          <div className="flex items-center gap-2 px-3 py-2 bg-emerald-500/15 border border-emerald-500/25 rounded-lg text-emerald-600 dark:text-emerald-400 text-xs select-none text-left">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shrink-0 animate-ping animate-duration-1000" />
            <span className="font-semibold text-[11px] leading-tight text-left">
              New Model Deployed (v2)!
            </span>
          </div>
        )}

        <button
          onClick={toggleTheme}
          className="flex items-center gap-2.5 px-2.5 py-2 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground text-left w-full transition-colors"
        >
          {theme === "light" ? (
            <>
              <Moon className="w-4 h-4" />
              <span className="text-[13px]">Dark Mode</span>
            </>
          ) : (
            <>
              <Sun className="w-4 h-4 text-amber-500" />
              <span className="text-[13px]">Light Mode</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
