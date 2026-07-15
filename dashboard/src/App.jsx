import React, { useState, useEffect } from "react";
import {
  Menu,
  Activity,
  Shield,
  ShieldCheck,
  ShieldAlert,
  Moon,
  Sun,
  X,
  Play
} from "lucide-react";
import { SidebarNav } from "./components/SidebarNav";
import TransactionFeed from "./components/TransactionFeed";
import ShapExplainer from "./components/ShapExplainer";
import ModelMetrics from "./components/ModelMetrics";
import DriftMonitor from "./components/DriftMonitor";
import SystemStatus from "./components/SystemStatus";
import { API_URL } from "./config";

export default function App() {
  const [activeView, setActiveView] = useState("feed");
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState("light");

  // Endpoint serving states
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [driftStatus, setDriftStatus] = useState(null);
  const [driftHistory, setDriftHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  // Initialize theme on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") || "light";
    setTheme(savedTheme);
    if (savedTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    localStorage.setItem("theme", nextTheme);
    if (nextTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const [demoActive, setDemoActive] = useState(false);
  const [demoStatus, setDemoStatus] = useState(null);
  const [autoRetrain, setAutoRetrain] = useState(true);

  // Auto-retrain loop when drift threshold (0.05) is exceeded
  useEffect(() => {
    if (demoActive && autoRetrain && demoStatus) {
      if (demoStatus.drift_score > 0.05 && demoStatus.stage !== 4 && demoStatus.stage !== 5) {
        const timer = setTimeout(() => {
          handleSetDemoStage(4);
        }, 1500);
        return () => clearTimeout(timer);
      }
    }
  }, [demoActive, autoRetrain, demoStatus?.drift_score, demoStatus?.stage]);

  // Background polling for service health & performance metrics
  useEffect(() => {
    fetchSystemData(); // Initial load
    const intervalTime = demoActive ? 1000 : 5000;
    const interval = setInterval(fetchSystemData, intervalTime);
    return () => clearInterval(interval);
  }, [demoActive]);

  const fetchSystemData = async () => {
    const host = API_URL;
    
    // Fetch demo status if active
    if (demoActive) {
      try {
        const res = await fetch(`${host}/demo/status`);
        if (res.ok) {
          const data = await res.json();
          setDemoStatus(data);
          if (!data.is_active) {
            setDemoActive(false);
          }
        }
      } catch {}
    }
    
    // 1. Health checks
    try {
      const res = await fetch(`${host}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth(null);
      }
    } catch {
      setHealth(null);
    }

    // 2. Metrics & registry
    try {
      const res = await fetch(`${host}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch {}

    try {
      const res = await fetch(`${host}/model/info`);
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch {}

    // 3. Drift statuses
    try {
      const res = await fetch(`${host}/drift`);
      if (res.ok) {
        const data = await res.json();
        setDriftStatus(data);
      }
    } catch {}

    try {
      const res = await fetch(`${host}/drift/history`);
      if (res.ok) {
        const data = await res.json();
        setDriftHistory(data);
      }
    } catch {}
  };

  const handleTriggerRetrain = async () => {
    try {
      const res = await fetch(`${API_URL}/retrain`, {
        method: "POST"
      });
      if (res.ok) {
        const data = await res.json();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  };

  const handleRunDemo = async () => {
    try {
      const res = await fetch(`${API_URL}/demo/start`, {
        method: "POST"
      });
      if (res.ok) {
        setDemoActive(true);
        setTimeout(fetchSystemData, 100);
      }
    } catch (err) {
      console.error("Failed to start demo", err);
    }
  };

  const handleSetDemoStage = async (stage) => {
    try {
      const res = await fetch(`${API_URL}/demo/stage?stage=${stage}`, {
        method: "POST"
      });
      if (res.ok) {
        fetchSystemData();
      }
    } catch (err) {
      console.error("Failed to set demo stage", err);
    }
  };

  const handleStopDemo = async () => {
    try {
      const res = await fetch(`${API_URL}/demo/stop`, {
        method: "POST"
      });
      if (res.ok) {
        setDemoActive(false);
        setDemoStatus(null);
        setTimeout(fetchSystemData, 100);
      }
    } catch (err) {
      console.error("Failed to stop demo", err);
    }
  };

  const handleSetCustomMetrics = async (params) => {
    try {
      const queryParams = new URLSearchParams();
      if (params.drift_score !== undefined) queryParams.append("drift_score", params.drift_score);
      if (params.accuracy !== undefined) queryParams.append("accuracy", params.accuracy);
      if (params.confidence !== undefined) queryParams.append("confidence", params.confidence);
      if (params.fraud_detection_rate !== undefined) queryParams.append("fraud_detection_rate", params.fraud_detection_rate);
      if (params.status !== undefined) queryParams.append("status", params.status);

      const res = await fetch(`${API_URL}/demo/custom?${queryParams.toString()}`, {
        method: "POST"
      });
      if (res.ok) {
        fetchSystemData();
      }
    } catch (err) {
      console.error("Failed to set custom demo metrics", err);
    }
  };

  // Map view IDs to display titles
  const viewTitles = {
    feed: "Live Transaction Feed",
    metrics: "Serving Performance Metrics",
    drift: "Feature Data Drift Monitor",
    system: "Servicing System Control"
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground transition-colors duration-200">
      
      {/* Sidebar Nav Component */}
      <SidebarNav
        activeView={activeView}
        onViewChange={(view) => {
          setActiveView(view);
          setSelectedTransaction(null);
        }}
        isSidebarOpen={isSidebarOpen}
        setSidebarOpen={setSidebarOpen}
        theme={theme}
        toggleTheme={toggleTheme}
        systemStatus={driftStatus}
        modelInfo={modelInfo}
        demoActive={demoActive}
        demoStatus={demoStatus}
        onRunDemo={handleRunDemo}
        onStopDemo={handleStopDemo}
        onSetDemoStage={handleSetDemoStage}
        onSetCustomMetrics={handleSetCustomMetrics}
        autoRetrain={autoRetrain}
        setAutoRetrain={setAutoRetrain}
      />

      {/* Main Panel Content Container */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        
        {/* Main top header bar */}
        <header className="h-14 border-b border-border/80 flex items-center px-4 justify-between bg-card shrink-0 select-none">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!isSidebarOpen)}
              className="p-1.5 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              title={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-xs text-muted-foreground font-medium">
              <span className="truncate">FraudShield Main</span>
              <span>/</span>
              <span className="font-semibold text-foreground truncate">
                {viewTitles[activeView]}
              </span>
            </div>
          </div>

          {/* Connection health indicators in header */}
          <div className="flex items-center gap-4">
            {demoActive ? (
              <div className="flex items-center gap-1 bg-muted/45 border border-border/80 p-0.5 rounded-lg">
                {[
                  { stage: 1, label: "Healthy" },
                  { stage: 2, label: "Drift" },
                  { stage: 3, label: "Alert" },
                  { stage: 4, label: "Retrain" },
                  { stage: 5, label: "Recover" }
                ].map((s) => (
                  <button
                    key={s.stage}
                    onClick={() => handleSetDemoStage(s.stage)}
                    className={`px-2 py-1 rounded-md text-[10px] font-bold transition-all cursor-pointer ${
                      demoStatus?.stage === s.stage
                        ? "bg-indigo-600 text-white shadow-xs"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/65"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
                <div className="h-4 w-px bg-border/80 mx-1" />
                <button
                  onClick={handleStopDemo}
                  className="p-1 rounded-md text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 transition-colors cursor-pointer"
                  title="Stop Demo"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={handleRunDemo}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold shadow-xs select-none transition-all duration-150 shrink-0 bg-indigo-600 text-white hover:bg-indigo-500 hover:shadow-md cursor-pointer"
              >
                <Play className="w-3 h-3 fill-current" />
                <span>Run Demo Mode</span>
              </button>
            )}

            <div className="hidden sm:flex items-center gap-2 border border-border/80 px-2.5 py-1 rounded-lg bg-background text-xs">
              <div
                className={`w-2 h-2 rounded-full ${
                  health ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                }`}
              />
              <span className="text-muted-foreground font-medium">
                Serving Node:{" "}
                <span className="text-foreground font-bold">
                  {health ? "ONLINE" : "OFFLINE"}
                </span>
              </span>
            </div>

            {/* Micro details panel in header */}
            <div className="hidden md:flex flex-col text-right leading-none select-none">
              <span className="text-[10px] text-muted-foreground font-semibold uppercase tracking-wider">
                Active Classifier
              </span>
              <span className="text-[11px] font-bold font-mono text-foreground mt-0.5">
                {modelInfo?.model_version || "local_v1"}
              </span>
            </div>
          </div>
        </header>

        {/* View swap frame area */}
        <main className="flex-1 overflow-y-auto bg-background/50 p-6">
          {activeView === "feed" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full items-start">
              {/* Transactions list taking 2 cols */}
              <div className="lg:col-span-2">
                <TransactionFeed
                  onSelectTransaction={setSelectedTransaction}
                  selectedTransaction={selectedTransaction}
                />
              </div>
              {/* SHAP Explainer taking 1 col */}
              <div className="lg:col-span-1">
                <ShapExplainer
                  shapExplanation={selectedTransaction?.shap_explanation}
                />
              </div>
            </div>
          )}

          {activeView === "metrics" && (
            <ModelMetrics metrics={metrics} modelInfo={modelInfo} />
          )}

          {activeView === "drift" && (
            <DriftMonitor
              driftStatus={driftStatus}
              driftHistory={driftHistory}
            />
          )}

          {activeView === "system" && (
            <SystemStatus
              health={health}
              modelInfo={modelInfo}
              onTriggerRetrain={handleTriggerRetrain}
              demoActive={demoActive}
              demoStatus={demoStatus}
            />
          )}
        </main>
      </div>
    </div>
  );
}

