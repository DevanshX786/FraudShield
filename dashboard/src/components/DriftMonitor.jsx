import React from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Calendar,
  AlertTriangle,
  Info
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import { ChartContainer, ChartTooltipContent } from "./ui/chart";

export default function DriftMonitor({ driftStatus, driftHistory }) {
  const driftDetected = driftStatus?.drift_detected || false;
  const features = driftStatus?.features || [];

  // Group flat drift history array by date/run
  const chartData = React.useMemo(() => {
    if (!driftHistory || driftHistory.length === 0) {
      // Fallback baseline trend data
      return [
        { date: "Day 1", amount: 0.02, velocity_1hr: 0.01, intensity: 0.1 },
        { date: "Day 2", amount: 0.04, velocity_1hr: 0.02, intensity: 0.3 },
        { date: "Day 3", amount: 0.11, velocity_1hr: 0.05, intensity: 0.5 },
        { date: "Day 4", amount: 0.23, velocity_1hr: 0.14, intensity: 0.7 }
      ];
    }

    const groups = {};
    driftHistory.forEach((r) => {
      const dateStr = new Date(r.run_date).toLocaleDateString();
      if (!groups[dateStr]) {
        groups[dateStr] = {
          date: dateStr,
          intensity: r.drift_intensity !== undefined ? r.drift_intensity : 0.1
        };
      }
      groups[dateStr][r.feature_name] = r.drift_score;
    });

    // Convert to sorted array
    return Object.values(groups).reverse();
  }, [driftHistory]);

  // Extract all feature names from history for legend/lines dynamic generation
  const activeFeatures = React.useMemo(() => {
    if (!driftHistory || driftHistory.length === 0) {
      return ["amount", "velocity_1hr"];
    }
    return Array.from(new Set(driftHistory.map((h) => h.feature_name))).slice(0, 5);
  }, [driftHistory]);

  const lineColors = ["#6366f1", "#0ea5e9", "#f43f5e", "#10b981", "#f59e0b"];

  const chartConfig = {
    intensity: { label: "Drift Intensity", color: "#6366f1" }
  };
  activeFeatures.forEach((feat, idx) => {
    chartConfig[feat] = { label: feat, color: lineColors[idx % lineColors.length] };
  });

  return (
    <div className="flex flex-col gap-6">
      {/* Overall Status Banner */}
      <div
        className={`flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-xl border shadow-xs text-left ${
          driftDetected
            ? "bg-rose-500/10 border-rose-500/20 text-rose-800 dark:text-rose-400"
            : "bg-emerald-500/10 border-emerald-500/20 text-emerald-800 dark:text-emerald-400"
        }`}
      >
        <div className="flex items-start gap-4">
          <div
            className={`p-2.5 rounded-lg border ${
              driftDetected
                ? "bg-rose-500/10 border-rose-500/20"
                : "bg-emerald-500/10 border-emerald-500/20"
            }`}
          >
            {driftDetected ? (
              <ShieldAlert className="w-5 h-5" />
            ) : (
              <ShieldCheck className="w-5 h-5" />
            )}
          </div>
          <div>
            <h3 className="text-sm font-bold">
              {driftDetected ? "Warning: Statistical Data Drift Detected" : "System Data State Stable"}
            </h3>
            <p className="text-xs text-muted-foreground/80 mt-0.5 max-w-xl leading-normal">
              {driftDetected
                ? "The distribution of incoming transaction features has shifted significantly from baseline training parameters. System retraining is recommended."
                : "Incoming transaction profiles match training parameters within acceptable statistical variations (KS & Chi-Square test p-values > 0.05)."}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 md:border-l border-current/20 md:pl-5 pr-1 font-mono text-xs">
          <Calendar className="w-4 h-4 shrink-0" />
          <span>Checked: {driftStatus?.run_date ? new Date(driftStatus.run_date).toLocaleDateString() : "Just now"}</span>
        </div>
      </div>

      {/* Feature statistical test scores list */}
      <div className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left">
        <div className="mb-4">
          <h3 className="text-[14px] font-bold text-foreground">Statistical Feature Testing</h3>
          <span className="text-[11px] text-muted-foreground">
            Test results comparing incoming batch variables vs baseline distribution parameters
          </span>
        </div>

        {features.length === 0 ? (
          <div className="text-center py-8 border border-dashed rounded-lg border-border/60">
            <Info className="w-6 h-6 text-muted-foreground/45 mx-auto mb-1.5" />
            <p className="text-xs text-muted-foreground">No drift metrics computed.</p>
            <span className="text-[10px] text-muted-foreground/60">
              Run statistical drift evaluation checks first.
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feat) => {
              const isDrifted = feat.drift_detected;
              return (
                <div
                  key={feat.name}
                  className={`flex flex-col p-4 bg-card border rounded-lg shadow-xs ${
                    isDrifted
                      ? "border-rose-500/25 bg-rose-500/[0.02]"
                      : "border-border/80 hover:border-border"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold font-mono text-foreground">
                      {feat.name}
                    </span>
                    <span
                      className={`inline-flex px-1.5 py-0.5 rounded text-[9px] font-bold border ${
                        isDrifted
                          ? "bg-rose-500/10 text-rose-500 border-rose-500/20"
                          : "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
                      }`}
                    >
                      {isDrifted ? "DRIFTED" : "STABLE"}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-border/40 text-[11px]">
                    <div className="flex flex-col text-left">
                      <span className="text-muted-foreground">P-Value</span>
                      <span className={`font-mono font-bold ${isDrifted ? "text-rose-500" : "text-foreground"}`}>
                        {feat.p_value?.toFixed(5) || "0.00000"}
                      </span>
                    </div>
                    <div className="flex flex-col text-left">
                      <span className="text-muted-foreground">Drift Score</span>
                      <span className="font-mono text-foreground font-semibold">
                        {feat.drift_score?.toFixed(3) || "0.000"}
                      </span>
                    </div>
                  </div>

                  {/* Warning message if drifted */}
                  {isDrifted && (
                    <div className="mt-3 flex items-center gap-1.5 text-[9px] text-rose-500/80 font-medium leading-none">
                      <AlertTriangle className="w-3 h-3 text-rose-500" />
                      <span>shift detected (p &lt; 0.05)</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Historical Drift intensity graphs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Drift Score Heatmap timeline */}
        <div className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left">
          <div className="mb-6">
            <h3 className="text-sm font-bold text-foreground">Feature Drift Scores</h3>
            <span className="text-[10px] text-muted-foreground">
              Kolmogorov-Smirnov & Chi-Square test statistics progression
            </span>
          </div>
          <div className="h-[250px] w-full">
            <ChartContainer config={chartConfig} className="aspect-auto h-full w-full">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border/40" />
                <XAxis
                  dataKey="date"
                  tickLine={false}
                  axisLine={false}
                  dy={10}
                  minTickGap={50}
                  className="fill-muted-foreground font-mono text-[9px]"
                />
                <YAxis
                  domain={[0.0, 1.0]}
                  tickLine={false}
                  axisLine={false}
                  dx={-10}
                  className="fill-muted-foreground font-mono text-[9px]"
                />
                <Tooltip content={<ChartTooltipContent />} />
                {activeFeatures.map((feat, idx) => (
                  <Line
                    key={feat}
                    type="monotone"
                    dataKey={feat}
                    stroke={lineColors[idx % lineColors.length]}
                    strokeWidth={2}
                    dot={{ strokeWidth: 1.5, r: 2.5 }}
                    name={feat}
                  />
                ))}
                <Legend verticalAlign="top" height={36} className="text-[10px]" />
              </LineChart>
            </ChartContainer>
          </div>
        </div>

        {/* Drift Simulation Intensity Progression */}
        <div className="p-5 bg-card border border-border/80 rounded-xl shadow-xs text-left">
          <div className="mb-6">
            <h3 className="text-sm font-bold text-foreground">Simulation Drift Intensity</h3>
            <span className="text-[10px] text-muted-foreground">
              Visualizes the simulated shift parameter in config.yaml over time
            </span>
          </div>
          <div className="h-[250px] w-full">
            <ChartContainer config={chartConfig} className="aspect-auto h-full w-full">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border/40" />
                <XAxis
                  dataKey="date"
                  tickLine={false}
                  axisLine={false}
                  dy={10}
                  minTickGap={50}
                  className="fill-muted-foreground font-mono text-[9px]"
                />
                <YAxis
                  domain={[0.0, 1.0]}
                  tickLine={false}
                  axisLine={false}
                  dx={-10}
                  className="fill-muted-foreground font-mono text-[9px]"
                />
                <Tooltip content={<ChartTooltipContent />} />
                <Line
                  type="monotone"
                  dataKey="intensity"
                  stroke="var(--color-intensity)"
                  strokeWidth={2.5}
                  dot={{ strokeWidth: 1.5, r: 3 }}
                  name="Drift Intensity"
                />
                <Legend verticalAlign="top" height={36} className="text-[10px]" />
              </LineChart>
            </ChartContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
