import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, Trash2, Search, HelpCircle, ShieldAlert, Sparkles } from "lucide-react";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell
} from "./ui/table";
import { API_URL } from "../config";

// Helper to generate a realistic mock transaction payload for the API
let txCounter = 1000;
function generateMockTransaction() {
  txCounter += 1;
  const cardNetworks = ["visa", "mastercard", "discover", "american express"];
  const cardTypes = ["debit", "credit"];
  const productCodes = ["W", "H", "C", "S", "R"];
  const emailDomains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "anonymous.com", "aol.com"];
  const deviceTypes = ["desktop", "mobile", "tablet", "unknown"];

  const isHighRisk = Math.random() < 0.15;
  
  let amt = isHighRisk 
    ? Math.round(150 + Math.random() * 850) 
    : Math.round(5 + Math.random() * 145);

  let productCd = productCodes[Math.floor(Math.random() * productCodes.length)];
  let card4 = cardNetworks[Math.floor(Math.random() * cardNetworks.length)];
  let pEmail = emailDomains[Math.floor(Math.random() * emailDomains.length)];

  return {
    transaction_id: `TX_${Date.now()}_${txCounter}`,
    transaction_amt: amt,
    product_cd: productCd,
    card1: Math.floor(1000 + Math.random() * 18000),
    card2: Math.floor(100 + Math.random() * 500),
    card3: Math.random() < 0.85 ? 150 : 185,
    card4: card4,
    card5: Math.random() < 0.9 ? 226 : 117,
    card6: cardTypes[Math.floor(Math.random() * cardTypes.length)],
    addr1: Math.random() < 0.8 ? 321 : 126,
    P_emaildomain: pEmail,
    R_emaildomain: Math.random() < 0.2 ? "gmail.com" : "unknown",
    DeviceType: deviceTypes[Math.floor(Math.random() * deviceTypes.length)],
    transaction_dt: Math.floor(Date.now() / 1000),
    is_fraud_ground_truth: isHighRisk ? 1 : 0
  };
}

export default function TransactionFeed({ onSelectTransaction, selectedTransaction }) {
  const [transactions, setTransactions] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [apiError, setApiError] = useState(null);
  


  const streamIntervalRef = useRef(null);

  // 1. Load initial recent transactions on mount
  useEffect(() => {
    fetchRecentTransactions();
    return () => stopStream();
  }, []);

  const fetchRecentTransactions = async () => {
    try {
      const res = await fetch(`${API_URL}/transactions/recent`);
      if (res.ok) {
        const data = await res.json();
        const formatted = data.map(tx => ({
          ...tx,
          shap_explanation: tx.shap_explanation || tx.shap_values || {}
        }));
        setTransactions(formatted);
        if (formatted.length > 0 && !selectedTransaction) {
          onSelectTransaction(formatted[0]);
        }
        setApiError(null);
      } else {
        setApiError("Failed to fetch recent transactions from serving API.");
      }
    } catch (err) {
      setApiError("Serving API is offline or database is unreachable.");
    }
  };



  // 2. Stream simulation triggers
  const startStream = () => {
    if (isStreaming) return;
    setIsStreaming(true);
    
    streamIntervalRef.current = setInterval(async () => {
      const mockTx = generateMockTransaction();
      try {
        const res = await fetch(`${API_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(mockTx)
        });
        
        if (res.ok) {
          const prediction = await res.json();

          const finalTx = {
            transaction_id: prediction.transaction_id,
            fraud_probability: prediction.fraud_probability,
            prediction: prediction.prediction,
            confidence: prediction.confidence,
            model_version: prediction.model_version,
            shap_explanation: prediction.shap_explanation || {},
            timestamp: new Date().toISOString(),
            amount: mockTx.transaction_amt,
            card_type: `${mockTx.card6} (${mockTx.card4})`,
            product_cd: mockTx.product_cd,
            card_network: mockTx.card4
          };
          
          setTransactions(prev => {
            const updated = [finalTx, ...prev].slice(0, 100);
            return updated;
          });
          onSelectTransaction(finalTx);
          setApiError(null);
        } else {
          setApiError("API responded with an error code on predict call.");
        }
      } catch (err) {
        setApiError("Serving API connection lost while streaming predictions.");
        stopStream();
      }
    }, 3000);
  };

  const stopStream = () => {
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    setIsStreaming(false);
  };

  const toggleStream = () => {
    if (isStreaming) {
      stopStream();
    } else {
      startStream();
    }
  };

  const clearFeed = () => {
    setTransactions([]);
    onSelectTransaction(null);
  };

  // 3. Search and filtering
  const filteredTransactions = transactions.filter(tx => {
    const idMatch = tx.transaction_id.toLowerCase().includes(searchTerm.toLowerCase());
    const predMatch = tx.prediction.toLowerCase().includes(searchTerm.toLowerCase());
    const cardMatch = tx.card_type && tx.card_type.toLowerCase().includes(searchTerm.toLowerCase());
    return idMatch || predMatch || cardMatch;
  });



  return (
    <div className="flex flex-col h-full bg-card border border-border/80 rounded-xl p-5 shadow-sm">
      {/* Feed Controls header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 mb-4 border-b border-border/60">
        <div>
          <h2 className="text-[15px] font-bold text-foreground flex items-center gap-2">
            Real-Time Transaction Feed
            {isStreaming && (
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
              </span>
            )}
          </h2>
          <span className="text-[11px] text-muted-foreground">
            Ingesting simulated payments & executing fraud prediction runs
          </span>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={toggleStream}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold shadow-xs transition-colors ${
              isStreaming
                ? "bg-slate-200 text-slate-800 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700"
                : "bg-primary text-primary-foreground hover:bg-primary/95"
            }`}
          >
            {isStreaming ? (
              <>
                <Pause className="w-3.5 h-3.5" />
                <span>Pause Feed</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>Start Stream</span>
              </>
            )}
          </button>

          <button
            onClick={fetchRecentTransactions}
            className="p-1.5 rounded-md border border-border hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
            title="Refresh list"
          >
            <Sparkles className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={clearFeed}
            className="p-1.5 rounded-md border border-border hover:bg-muted text-muted-foreground hover:text-destructive transition-colors"
            title="Clear list"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* API connection alerts */}
      {apiError && (
        <div className="mb-4 flex items-start gap-2.5 bg-rose-500/10 border border-rose-500/25 p-3 rounded-lg text-rose-600 dark:text-rose-400 text-xs">
          <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="text-left">
            <span className="font-semibold">Backend Connection Issue</span>
            <p className="text-[10px] mt-0.5 text-rose-500/80">{apiError}</p>
          </div>
        </div>
      )}



      {/* Search Input */}
      <div className="relative mb-4">
        <Search className="absolute left-3 top-2.5 w-3.5 h-3.5 text-muted-foreground/60" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Filter by ID, prediction, network..."
          className="w-full pl-9 pr-4 py-2 border border-border rounded-lg bg-muted/20 text-xs text-foreground placeholder:text-muted-foreground/50 outline-none focus:border-primary/80 focus:ring-1 focus:ring-primary/20 transition-colors"
        />
      </div>

      {/* Transaction Feed Table */}
      <div className="flex-1 overflow-y-auto max-h-[480px]">
        {filteredTransactions.length === 0 ? (
          <div className="py-12 text-center text-muted-foreground border border-dashed rounded-lg border-border/80">
            <p className="text-xs">No transactions in the feed</p>
            <span className="text-[10px] text-muted-foreground/60 block mt-1">
              Start the stream or refresh recent records.
            </span>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Transaction ID</TableHead>
                <TableHead>Type</TableHead>
                <TableHead className="text-right">Amount</TableHead>
                <TableHead>Risk Probability</TableHead>
                <TableHead>Prediction</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredTransactions.map((tx) => {
                const isSelected = selectedTransaction?.transaction_id === tx.transaction_id;
                
                // Color mapping for badge states
                let badgeColor = "bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border-emerald-500/20";
                if (tx.prediction === "FRAUD") {
                  badgeColor = "bg-rose-500/10 text-rose-500 dark:text-rose-400 border-rose-500/20";
                } else if (tx.prediction === "SUSPICIOUS") {
                  badgeColor = "bg-amber-500/10 text-amber-500 dark:text-amber-400 border-amber-500/20";
                }

                // Format Amount
                const txAmt = tx.amount || tx.transaction_amt || 0;

                return (
                  <TableRow
                    key={tx.transaction_id}
                    data-state={isSelected ? "selected" : undefined}
                    onClick={() => onSelectTransaction(tx)}
                    className="cursor-pointer"
                  >
                    <TableCell className="font-mono font-medium truncate max-w-[120px]" title={tx.transaction_id}>
                      {tx.transaction_id}
                    </TableCell>
                    <TableCell className="capitalize text-muted-foreground">
                      {tx.card_type || "debit"}
                    </TableCell>
                    <TableCell className="text-right font-mono font-medium text-foreground">
                      ${txAmt.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden shrink-0">
                          <div
                            className={`h-full rounded-full ${
                              tx.fraud_probability >= 0.5 ? "bg-rose-500" : tx.fraud_probability >= 0.3 ? "bg-amber-500" : "bg-emerald-500"
                            }`}
                            style={{ width: `${tx.fraud_probability * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {(tx.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border ${badgeColor}`}>
                        {tx.prediction}
                      </span>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}
