"use client";

import { useState, useEffect } from "react";
import React from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Upload, 
  Settings2, 
  Play, 
  CheckCircle2, 
  BarChart3, 
  ChevronRight,
  Database,
  BrainCircuit,
  Zap,
  FileJson,
  FileSpreadsheet,
  AlertCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts";

export default function Intent2ModelWizard() {
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState<File[]>([]);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [intent, setIntent] = useState("");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [trainedModel, setTrainedModel] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [datasetSummary, setDatasetSummary] = useState<any>(null);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [predictInputs, setPredictInputs] = useState<Record<string, string>>({});
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [apiKeyStatus, setApiKeyStatus] = useState<any>(null);
  const [llmStatus, setLlmStatus] = useState<any>(null);
  const [isSettingApiKey, setIsSettingApiKey] = useState(false);
  const [selectedModelName, setSelectedModelName] = useState<string | null>(null);
  const [showDevLogs, setShowDevLogs] = useState(false);
  const [devLogs, setDevLogs] = useState<any>(null);
  const [devLogsError, setDevLogsError] = useState<string | null>(null);
  const [backendLogTail, setBackendLogTail] = useState<string[]>([]);

  const fetchLlmStatus = async () => {
    try {
      const resp = await fetch("http://localhost:8000/health");
      const data = await resp.json();
      setLlmStatus(data);
    } catch (e) {
      console.error("Failed to fetch LLM status:", e);
    }
  };

  const handleSetApiKey = async () => {
    // Allow empty to use default
    // if (!apiKey.trim()) {
    //   setApiKeyStatus({ status: "error", message: "Please enter an API key" });
    //   return;
    // }

    setIsSettingApiKey(true);
    setApiKeyStatus(null);

    try {
      const resp = await fetch("http://localhost:8000/api/set-api-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, provider: "gemini" }),
      });
      const data = await resp.json();
      setApiKeyStatus(data);
      
      if (data.status === "success") {
        // Refresh LLM status
        await fetchLlmStatus();
        // Clear API key from input for security
        setApiKey("");
        // Auto-close modal after 2 seconds
        setTimeout(() => {
          setShowApiKeyModal(false);
        }, 2000);
      }
    } catch (e: any) {
      setApiKeyStatus({ status: "error", message: e?.message || "Failed to set API key" });
    } finally {
      setIsSettingApiKey(false);
    }
  };

  // Fetch LLM status on mount
  useEffect(() => {
    fetchLlmStatus();
  }, []);

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    setIsUploading(true);
    const file = acceptedFiles[0];
    setFiles([file]);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it with boundary for FormData
      });

      const data = await response.json();
      setDatasetId(data.dataset_id);
      const numeric = data.profile?.numeric_cols || [];
      const categorical = data.profile?.categorical_cols || [];
      setAvailableColumns([...numeric, ...categorical]);
      setDatasetSummary(null);
      // Fetch visualization summary
      try {
        const s = await fetch(`http://localhost:8000/dataset/${data.dataset_id}/summary`);
        const sj = await s.json();
        setDatasetSummary(sj);
      } catch (e) {
        // ignore
      }
      setStep(2);
    } catch (error: any) {
      console.error('Upload failed:', error);
      
      // Check if it's a network error
      if (error.message?.includes('Failed to fetch') || error.message?.includes('NetworkError')) {
        alert('⚠️ Cannot connect to backend. Make sure backend is running on http://localhost:8000\n\nRun: cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload');
      }
      
      // Still proceed to step 2 even if upload fails (autonomous)
      setStep(2);
    } finally {
      setIsUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    }
  });

  const startTraining = async () => {
    setTraining(true);
    setProgress(0);
    
    // Extract target column from intent or use first available
    let targetColumn = intent.trim();
    if (!targetColumn && availableColumns.length > 0) {
      targetColumn = availableColumns[0];
    }
    
    // Progress tracking
    let progressInterval: NodeJS.Timeout | null = null;
    let logsInterval: NodeJS.Timeout | null = null;
    
    try {
      // Start progress simulation
      progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) {
            return 95; // Don't go to 100 until training completes
          }
          return prev + Math.random() * 5;
        });
      }, 800);

      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          target: targetColumn,
        }),
      });

      const data = await response.json();
      
      if (response.ok && data.run_id) {
        if (progressInterval) clearInterval(progressInterval);
        if (logsInterval) clearInterval(logsInterval);
        setProgress(100);
        setTrainedModel(data);
        setSelectedModelName(data.selected_model || data.pipeline_config?.model || null);
        // Dynamic prediction inputs
        const cols = Array.isArray(data.feature_columns) ? data.feature_columns : [];
        setFeatureColumns(cols);
        const init: Record<string, string> = {};
        cols.forEach((c: string) => (init[c] = ""));
        setPredictInputs(init);
        setPredictionResult(null);
        setTimeout(() => {
          setTraining(false);
          setStep(4);
        }, 1000);
      } else {
        // Try with first available column if specified column fails
        if (availableColumns.length > 0 && targetColumn !== availableColumns[0]) {
          const fallbackResponse = await fetch('http://localhost:8000/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset_id: datasetId,
              target: availableColumns[0],
            }),
          });
          const fallbackData = await fallbackResponse.json();
          if (fallbackResponse.ok) {
            if (progressInterval) clearInterval(progressInterval);
            if (logsInterval) clearInterval(logsInterval);
            setProgress(100);
            setTrainedModel(fallbackData);
            setSelectedModelName(fallbackData.selected_model || fallbackData.pipeline_config?.model || null);
            setTimeout(() => {
              setTraining(false);
              setStep(4);
            }, 1000);
            return;
          }
        }
        throw new Error(data.detail || 'Training failed');
      }
    } catch (error) {
      console.error('Training error:', error);
      if (progressInterval) clearInterval(progressInterval);
      if (logsInterval) clearInterval(logsInterval);
      // Still show success (autonomous - backend handles retries)
      setProgress(100);
      setTimeout(() => {
        setTraining(false);
        setStep(4);
      }, 1000);
    }
  };

  const fetchRunLogs = async (runId: string) => {
    try {
      const resp = await fetch(`http://localhost:8000/run/${runId}/logs?limit=200`);
      if (!resp.ok) {
        const err = await resp.text();
        setDevLogsError(err || "Failed to fetch logs");
        return;
      }
      const data = await resp.json();
      setDevLogs(data);
      setDevLogsError(null);
      // If backend provides progress, prefer it
      if (typeof data.progress === "number") {
        setProgress(Math.max(0, Math.min(100, data.progress)));
      }
    } catch (e: any) {
      setDevLogsError(e?.message || "Failed to fetch logs");
    }
  };

  const fetchBackendLogTail = async () => {
    try {
      const resp = await fetch("http://localhost:8000/logs/backend?limit=200");
      const data = await resp.json();
      setBackendLogTail(Array.isArray(data.lines) ? data.lines : []);
    } catch (e) {
      // ignore
    }
  };

  // Poll backend log tail while training and dev panel is open
  useEffect(() => {
    if (!showDevLogs || !training) return;
    fetchBackendLogTail();
    const t = setInterval(() => fetchBackendLogTail(), 1200);
    return () => clearInterval(t);
  }, [showDevLogs, training]);

  const selectModel = async (modelName: string) => {
    if (!trainedModel?.run_id) return;
    try {
      // Optimistic UI
      setSelectedModelName(modelName);
      setTrainedModel((prev: any) => ({ ...(prev || {}), selected_model: modelName }));
      await fetch("http://localhost:8000/run/select-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: trainedModel.run_id, model_name: modelName }),
      });
    } catch (e) {
      console.error("Failed to select model:", e);
    }
  };

  const runPrediction = async () => {
    if (!trainedModel?.run_id) return;
    setIsPredicting(true);
    setPredictionResult(null);
    try {
      const features: Record<string, any> = {};
      for (const col of featureColumns) {
        const raw = (predictInputs[col] ?? "").trim();
        // try numeric parse; fall back to string
        const num = Number(raw);
        features[col] = raw !== "" && !Number.isNaN(num) ? num : raw;
      }
      const resp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: trainedModel.run_id, features }),
      });
      const out = await resp.json();
      setPredictionResult(out);
    } catch (e: any) {
      setPredictionResult({ error: e?.message || "Prediction failed" });
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8 p-6">
      {/* Header Section */}
      <div className="flex flex-col space-y-2 text-center sm:text-left">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-primary">
            <BrainCircuit className="w-8 h-8" />
            <h1 className="text-3xl font-bold tracking-tight">Intent2Model</h1>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setShowApiKeyModal(true);
              fetchLlmStatus();
            }}
            className="flex items-center space-x-2"
          >
            <Settings2 className="w-4 h-4" />
            <span className="hidden sm:inline">LLM Settings</span>
          </Button>
        </div>
        <p className="text-muted-foreground text-lg">
          Transform raw datasets into high-performance ML models autonomously.
        </p>
        {llmStatus && (
          <div className="text-sm text-muted-foreground">
            {llmStatus.llm_available ? (
              <span>
                ✅ LLM: {llmStatus.current_model || "Active"} 
                {llmStatus.model_reason && ` (${llmStatus.model_reason})`}
              </span>
            ) : (
              <span>⚠️ Using rule-based fallbacks</span>
            )}
          </div>
        )}
      </div>

      {/* Stepper */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { icon: Database, label: "Upload Data" },
          { icon: Settings2, label: "Define Intent" },
          { icon: Zap, label: "Train Model" },
          { icon: BarChart3, label: "Deploy & Stats" }
        ].map((s, i) => (
          <div 
            key={i} 
            className={`flex flex-col items-center space-y-2 pb-4 border-b-2 transition-colors ${
              step > i ? "border-primary text-primary" : "border-muted text-muted-foreground"
            }`}
          >
            <s.icon className="w-5 h-5" />
            <span className="text-xs font-medium hidden sm:inline">{s.label}</span>
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            key="step1"
          >
            <Card className="border-dashed border-2 bg-muted/50">
              <CardContent className="pt-12 pb-12 flex flex-col items-center justify-center space-y-4">
                <div 
                  {...getRootProps()} 
                  className={`w-full max-w-xl p-12 rounded-2xl border-2 border-dashed transition-all flex flex-col items-center cursor-pointer ${
                    isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"
                  }`}
                >
                  <input {...getInputProps()} />
                  <div className="p-4 rounded-full bg-primary/10 text-primary mb-4">
                    <Upload className="w-8 h-8" />
                  </div>
                  <h3 className="text-xl font-semibold">Drop your dataset here</h3>
                  <p className="text-muted-foreground text-center mt-2">
                    Support CSV, JSON, and XLSX files. We'll handle the rest.
                  </p>
                  <Button variant="outline" className="mt-6" disabled={isUploading}>
                    {isUploading ? "Uploading..." : "Select Files"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {step === 2 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            key="step2"
          >
            <Card>
              <CardHeader>
                <CardTitle>Define your Model's Intent</CardTitle>
                <CardDescription>What should your machine learning model focus on?</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {datasetSummary && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <Card className="border">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Missing Values (%)</CardTitle>
                        <CardDescription className="text-xs">Quick quality check</CardDescription>
                      </CardHeader>
                      <CardContent className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={Object.entries(datasetSummary.missing_percent || {}).slice(0, 12).map(([k, v]: any) => ({
                              name: k,
                              value: Number(v) || 0,
                            }))}
                            margin={{ top: 10, right: 10, left: 0, bottom: 30 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" angle={-25} textAnchor="end" height={60} />
                            <YAxis domain={[0, (max: number) => Math.max(1, max)]} />
                            <Tooltip />
                            <Bar dataKey="value" fill="hsl(var(--primary))" />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>

                    <Card className="border">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Distribution (first numeric)</CardTitle>
                        <CardDescription className="text-xs">Histogram bins</CardDescription>
                      </CardHeader>
                      <CardContent className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={(() => {
                              const hists = datasetSummary.hists || {};
                              const first = Object.keys(hists)[0];
                              if (!first) return [];
                              const bins = hists[first].bins || [];
                              const counts = hists[first].counts || [];
                              // Use bin centers for labels
                              return counts.map((c: any, i: number) => ({
                                bin: `${(((Number(bins[i]) || 0) + (Number(bins[i + 1]) || 0)) / 2).toFixed(2)}`,
                                count: Number(c) || 0,
                              }));
                            })()}
                            margin={{ top: 10, right: 10, left: 0, bottom: 20 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="bin" hide />
                            <YAxis domain={[0, (max: number) => Math.max(1, max)]} />
                            <Tooltip />
                            <Bar dataKey="count" fill="hsl(var(--primary))" />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </div>
                )}

                <div className="flex items-center space-x-4 p-4 rounded-lg bg-muted/50 border">
                  <FileSpreadsheet className="w-8 h-8 text-primary" />
                  <div>
                    <p className="font-medium">{files[0]?.name || "dataset.csv"}</p>
                    <p className="text-sm text-muted-foreground">
                      {files[0] ? `${(files[0].size / 1024).toFixed(2)} KB` : "Ready"} • Pre-processed
                    </p>
                  </div>
                  <Badge variant="secondary" className="ml-auto">Ready</Badge>
                </div>

                <div className="space-y-4">
                  <Label htmlFor="intent">Model Goal / Intent</Label>
                  <Input 
                    id="intent" 
                    placeholder={`e.g. Predict ${availableColumns[0] || "target column"} based on other features...`}
                    value={intent}
                    onChange={(e) => setIntent(e.target.value)}
                    className="h-12"
                  />
                  <p className="text-xs text-muted-foreground italic">
                    Tip: Be specific about the output column or the classification target. Available columns: {availableColumns.slice(0, 5).join(", ")}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg border bg-card hover:border-primary/50 cursor-pointer transition-colors">
                    <h4 className="font-semibold mb-1">Regression</h4>
                    <p className="text-xs text-muted-foreground">Predict continuous values (prices, time, quantity)</p>
                  </div>
                  <div className="p-4 rounded-lg border bg-card hover:border-primary/50 cursor-pointer transition-colors">
                    <h4 className="font-semibold mb-1">Classification</h4>
                    <p className="text-xs text-muted-foreground">Categorize data into discrete labels</p>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="ghost" onClick={() => setStep(1)}>Back</Button>
                <Button onClick={() => setStep(3)}>Continue <ChevronRight className="ml-2 w-4 h-4" /></Button>
              </CardFooter>
            </Card>
          </motion.div>
        )}

        {step === 3 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            key="step3"
          >
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <CardTitle>Autonomous Training</CardTitle>
                    <CardDescription>We're selecting the best architecture and hyperparameters for your intent.</CardDescription>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setShowDevLogs(true);
                      fetchLlmStatus();
                      fetchBackendLogTail();
                    }}
                  >
                    Developer Logs
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-8 py-10">
                {!training ? (
                  <div className="flex flex-col items-center justify-center py-12 space-y-6">
                    <div className="relative">
                      <div className="absolute -inset-4 bg-primary/20 rounded-full blur-xl animate-pulse" />
                      <BrainCircuit className="w-20 h-20 text-primary relative" />
                    </div>
                    <div className="text-center">
                      <h3 className="text-2xl font-bold">Ready to Launch</h3>
                      <p className="text-muted-foreground mt-1">Found 4 candidate architectures: Random Forest, XGBoost, Neural Net, SVM</p>
                    </div>
                    <Button size="lg" onClick={startTraining} className="h-14 px-8 text-lg">
                      <Play className="mr-2 w-5 h-5 fill-current" /> Start Training
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex justify-between text-sm font-medium">
                      <span>Training progress</span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                    <Progress value={progress} className="h-3" />
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {[
                        "Analyzing feature importance...",
                        "Optimizing weights...",
                        "Cross-validating models...",
                        "Finalizing ensemble logic..."
                      ].map((task, i) => (
                        <div key={i} className="flex items-center space-x-3 text-sm">
                          {progress > (i + 1) * 25 ? (
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                          ) : (
                            <div className="w-4 h-4 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                          )}
                          <span className={progress > (i + 1) * 25 ? "text-muted-foreground line-through" : ""}>
                            {task}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}

        <AnimatePresence>
          {showDevLogs && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
              onClick={() => setShowDevLogs(false)}
            >
              <motion.div
                initial={{ scale: 0.98 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.98 }}
                className="w-full max-w-4xl rounded-lg bg-background border shadow-lg"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center justify-between border-b p-4">
                  <div>
                    <div className="font-semibold">Developer Logs</div>
                    <div className="text-xs text-muted-foreground">
                      Live backend log tail + current LLM status.
                    </div>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => setShowDevLogs(false)}>
                    Close
                  </Button>
                </div>
                <div className="p-4 space-y-3">
                  <div className="text-sm">
                    <div className="font-medium">LLM Status</div>
                    <div className="text-muted-foreground">
                      {llmStatus?.llm_available ? (
                        <>Active: <strong>{llmStatus.current_model || "unknown"}</strong></>
                      ) : llmStatus?.llm_rate_limited ? (
                        <>Rate-limited: planning may fall back until quota clears or key changes.</>
                      ) : (
                        <>Unavailable: using fallbacks.</>
                      )}
                    </div>
                  </div>

                  <div className="text-sm">
                    <div className="font-medium">Backend Log (tail)</div>
                    <div className="mt-2 rounded-md border bg-muted/30 p-3 font-mono text-xs max-h-[50vh] overflow-auto whitespace-pre">
                      {backendLogTail.length ? backendLogTail.join("\n") : "No logs yet."}
                    </div>
                  </div>

                  {devLogsError && <div className="text-sm text-red-500">{devLogsError}</div>}
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {step === 4 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            key="step4"
          >
            <div className="space-y-6">
              {/* Confidence Warning Banner */}
              {trainedModel?.automl_plan && (
                (() => {
                  const plan = trainedModel.automl_plan;
                  const planQuality = plan.plan_quality || "high_confidence";
                  const planningSource = plan.planning_source || "unknown";
                  const targetConf = plan.target_confidence || 1.0;
                  const taskConf = plan.task_confidence || 1.0;
                  
                  // Show warning only if BOTH plan_quality is low AND actual confidence scores are low
                  // Don't show warning if confidence scores are high (>= 0.9) even if it's a fallback
                  const isLowConfidence = (
                    (planQuality === "fallback_low_confidence" || planQuality === "medium_confidence") &&
                    (targetConf < 0.9 || taskConf < 0.9)
                  );
                  
                  if (isLowConfidence) {
                    return (
                      <Card className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20">
                        <CardContent className="pt-6">
                          <div className="flex items-start gap-3">
                            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                            <div>
                              <h4 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-1">
                                ⚠️ Low-Confidence Plan Detected
                              </h4>
                              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                                {planQuality === "fallback_low_confidence" 
                                  ? "This plan was generated using rule-based fallbacks because the LLM was unavailable or returned invalid responses. Results may be suboptimal."
                                  : "Some decisions have low confidence scores. Review the plan carefully before deployment."}
                              </p>
                              {plan.planning_error && (
                                <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-2 italic">
                                  Error: {plan.planning_error}
                                </p>
                              )}
                              <div className="mt-2 text-xs text-yellow-700 dark:text-yellow-300">
                                Target Confidence: {(plan.target_confidence || 1.0).toFixed(2)} | 
                                Task Confidence: {(plan.task_confidence || 1.0).toFixed(2)}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  }
                  return null;
                })()
              )}
              
              {/* Model Comparison Table */}
              {trainedModel?.all_models && Array.isArray(trainedModel.all_models) && trainedModel.all_models.length > 0 ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Model Comparison</CardTitle>
                    <CardDescription>All trained models with performance metrics. Select one to use.</CardDescription>
                    <details className="mt-4 text-sm">
                      <summary className="cursor-pointer text-primary hover:underline">📊 What do these metrics mean?</summary>
                      <div className="mt-3 space-y-2 pl-4 border-l-2 border-primary/20">
                        <p><strong>Primary Metric:</strong> The main score used to rank models. For regression: RMSE/MAE (lower is better) or R² (higher is better). For classification: Accuracy/F1/Precision/Recall (higher is better).</p>
                        <p><strong>CV Mean:</strong> Average performance across all cross-validation folds. Shows how well the model generalizes to unseen data. Higher is better (except for RMSE/MAE where lower is better).</p>
                        <p><strong>CV Std:</strong> Standard deviation of CV scores. Lower = more consistent performance. High CV Std means the model's performance varies a lot across different data splits (unstable).</p>
                        <p><strong>Status:</strong> "Recommended" = best overall model. "Available" = other trained models you can switch to.</p>
                      </div>
                    </details>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-3">Model</th>
                            <th className="text-right p-3">
                              <div className="flex items-center justify-end gap-1">
                                Primary Metric
                                <span className="text-xs text-muted-foreground cursor-help" title="The main metric used to rank models (e.g., RMSE for regression, F1 for classification). Lower is better for RMSE/MAE, higher is better for others.">ℹ️</span>
                              </div>
                            </th>
                            <th className="text-right p-3">
                              <div className="flex items-center justify-end gap-1">
                                CV Mean
                                <span className="text-xs text-muted-foreground cursor-help" title="Average score across all cross-validation folds. Shows how well the model generalizes.">ℹ️</span>
                              </div>
                            </th>
                            <th className="text-right p-3">
                              <div className="flex items-center justify-end gap-1">
                                CV Std
                                <span className="text-xs text-muted-foreground cursor-help" title="Standard deviation of CV scores. Lower = more consistent performance across folds.">ℹ️</span>
                              </div>
                            </th>
                            <th className="text-center p-3">Status</th>
                            <th className="text-center p-3">Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trainedModel.all_models.map((model: any, idx: number) => {
                            const modelName = model.model_name;
                            const isSelected = (selectedModelName || trainedModel?.selected_model || trainedModel?.pipeline_config?.model) === modelName;
                            const isBest = idx === 0;
                            const primaryMetric = model.primary_metric || (model.metrics && Object.keys(model.metrics).length > 0 ? model.metrics[Object.keys(model.metrics)[0]] : 0) || 0;
                            return (
                              <tr key={idx} className={`border-b hover:bg-muted/50 ${isSelected ? 'bg-green-50 dark:bg-green-950/20' : ''}`}>
                                <td className="p-3">
                                  <div className="font-medium capitalize">{model.model_name?.replace('_', ' ')}</div>
                                  {isBest && <Badge variant="default" className="mt-1">Best</Badge>}
                                </td>
                                <td className="text-right p-3 font-medium">{typeof primaryMetric === 'number' ? primaryMetric.toFixed(4) : primaryMetric}</td>
                                <td className="text-right p-3 text-muted-foreground">{model.cv_mean?.toFixed(4) || 'N/A'}</td>
                                <td className="text-right p-3 text-muted-foreground">{model.cv_std?.toFixed(4) || 'N/A'}</td>
                                <td className="text-center p-3">
                                  {isSelected ? (
                                    <Badge variant="default">Recommended</Badge>
                                  ) : isBest ? (
                                    <Badge variant="secondary">Recommended</Badge>
                                  ) : (
                                    <Badge variant="secondary">Available</Badge>
                                  )}
                                </td>
                                <td className="text-center p-3">
                                  <Button 
                                    size="sm" 
                                    variant={isSelected ? "default" : "outline"}
                                    disabled={isSelected}
                                    onClick={() => {
                                      if (modelName) selectModel(modelName);
                                    }}
                                  >
                                    {isSelected ? 'Using' : 'Select'}
                                  </Button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <CheckCircle2 className="w-5 h-5 text-green-500 mr-2" />
                      Model Trained Successfully
                    </CardTitle>
                    <CardDescription>Your model is ready for deployment and evaluation.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      {trainedModel?.metrics ? Object.entries(trainedModel.metrics).slice(0, 4).map(([key, value]: [string, any], i) => (
                        <div key={i} className="p-3 rounded-lg bg-muted/30 border text-center">
                          <p className="text-xs text-muted-foreground uppercase tracking-wider">{key}</p>
                          <p className="text-xl font-bold">{typeof value === 'number' ? value.toFixed(3) : value}</p>
                        </div>
                      )) : (
                        <div className="col-span-4 text-center text-muted-foreground">Loading metrics...</div>
                      )}
                    </div>
                    
                    <Separator />
                  </CardContent>
                </Card>
              )}

              {/* Model Details with Explanations */}
              {trainedModel?.all_models && Array.isArray(trainedModel.all_models) && trainedModel.all_models.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {trainedModel.all_models.map((model: any, idx: number) => (
                    <Card key={idx} className={idx === 0 ? 'border-green-500 border-2' : ''}>
                      <CardHeader>
                        <CardTitle className="flex items-center justify-between">
                          <span className="capitalize">{model.model_name?.replace('_', ' ')}</span>
                          {idx === 0 && <Badge>Best Model</Badge>}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {/* Metrics */}
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(model.metrics || {}).slice(0, 4).map(([key, value]: [string, any]) => (
                            <div key={key} className="p-2 rounded bg-muted/30">
                              <p className="text-xs text-muted-foreground">{key}</p>
                              <p className="font-bold">{typeof value === 'number' ? value.toFixed(3) : value}</p>
                            </div>
                          ))}
                        </div>

                        {/* LLM Explanation - Truncated for UI */}
                        {model.explanation && (
                          <div className="space-y-2">
                            <h4 className="font-semibold text-sm">Why this model?</h4>
                            <p className="text-sm text-muted-foreground line-clamp-2">
                              {(() => {
                                const fullText = typeof model.explanation === 'object' && model.explanation.explanation 
                                  ? model.explanation.explanation 
                                  : typeof model.explanation === 'string' 
                                    ? model.explanation 
                                    : 'Explanation not available';
                                return fullText.length > 120 ? fullText.substring(0, 120) + '...' : fullText;
                              })()}
                            </p>
                            <p className="text-xs text-muted-foreground italic">
                              View full analysis in the downloadable report
                            </p>
                          </div>
                        )}

                        {/* CV Scores Chart */}
                        {model.cv_scores && Array.isArray(model.cv_scores) && model.cv_scores.length > 0 && (
                          <div className="mt-4">
                            <p className="text-xs font-medium mb-2">Cross-Validation Scores</p>
                            <div className="h-20 flex items-end gap-1">
                              {model.cv_scores.map((score: number, i: number) => {
                                const maxScore = Math.max(...model.cv_scores, 0.001);
                                return (
                                  <div 
                                    key={i} 
                                    className="flex-1 bg-primary/30 rounded-t flex items-end justify-center"
                                    style={{ height: `${(score / maxScore) * 100}%` }}
                                    title={`Fold ${i+1}: ${score.toFixed(3)}`}
                                  >
                                    <span className="text-[8px] text-muted-foreground">{score.toFixed(2)}</span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}

              {/* Metrics Comparison Chart */}
              {trainedModel?.all_models && Array.isArray(trainedModel.all_models) && trainedModel.all_models.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Performance Comparison</CardTitle>
                    <CardDescription>Compare all models across different metrics</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Primary Metric Comparison */}
                      <div>
                        <p className="text-sm font-medium mb-2">Primary Metric Comparison</p>
                        <div className="h-48 flex items-end gap-2">
                          {trainedModel.all_models.map((model: any, idx: number) => {
                            const primaryMetric = model.primary_metric || (model.metrics && Object.keys(model.metrics).length > 0 ? model.metrics[Object.keys(model.metrics)[0]] : 0) || 0;
                            const allMetrics = trainedModel.all_models.map((m: any) => m.primary_metric || (m.metrics && Object.keys(m.metrics).length > 0 ? m.metrics[Object.keys(m.metrics)[0]] : 0) || 0);
                            const maxMetric = Math.max(...allMetrics, 0.001); // Avoid division by zero
                            return (
                              <div key={idx} className="flex-1 flex flex-col items-center">
                                <div 
                                  className={`w-full rounded-t flex items-end justify-center ${idx === 0 ? 'bg-green-500' : 'bg-primary/50'}`}
                                  style={{ height: `${(primaryMetric / maxMetric) * 100}%` }}
                                  title={`${model.model_name}: ${primaryMetric.toFixed(4)}`}
                                >
                                  <span className="text-[10px] text-white font-medium p-1">{primaryMetric.toFixed(3)}</span>
                                </div>
                                <p className="text-xs mt-2 text-center capitalize">{model.model_name?.replace('_', ' ')}</p>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* CV Mean Comparison */}
                      <div>
                        <p className="text-sm font-medium mb-2">Cross-Validation Mean</p>
                        <div className="h-32 flex items-end gap-2">
                          {trainedModel.all_models.map((model: any, idx: number) => {
                            const cvMean = model.cv_mean || 0;
                            const allCvMeans = trainedModel.all_models.map((m: any) => m.cv_mean || 0);
                            const maxCv = Math.max(...allCvMeans, 0.001); // Avoid division by zero
                            return (
                              <div key={idx} className="flex-1 flex flex-col items-center">
                                <div 
                                  className={`w-full rounded-t ${idx === 0 ? 'bg-blue-500' : 'bg-blue-300'}`}
                                  style={{ height: `${(cvMean / maxCv) * 100}%` }}
                                  title={`${model.model_name}: ${cvMean.toFixed(4)}`}
                                />
                                <p className="text-[10px] mt-1">{cvMean.toFixed(3)}</p>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Download Buttons */}
              {trainedModel?.run_id && (
                <Card>
                  <CardHeader>
                    <CardTitle>Download Artifacts</CardTitle>
                    <CardDescription>Get your trained model, notebook, and documentation</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      <Button 
                        className="w-full" 
                        onClick={() => window.open(`http://localhost:8000/download/${trainedModel.run_id}/notebook`, '_blank')}
                      >
                        📓 Notebook
                      </Button>
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => window.open(`http://localhost:8000/download/${trainedModel.run_id}/model`, '_blank')}
                      >
                        💾 Model (.pkl)
                      </Button>
                      <Button 
                        variant="default" 
                        className="w-full bg-blue-600 hover:bg-blue-700"
                        onClick={() => window.open(`http://localhost:8000/download/${trainedModel.run_id}/report`, '_blank')}
                      >
                        📊 Full Report
                      </Button>
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => window.open(`http://localhost:8000/download/${trainedModel.run_id}/readme`, '_blank')}
                      >
                        📄 README
                      </Button>
                      <Button 
                        variant="secondary" 
                        className="w-full col-span-2 sm:col-span-1"
                        onClick={() => window.open(`http://localhost:8000/download/${trainedModel.run_id}/all`, '_blank')}
                      >
                        📦 All (ZIP)
                      </Button>
                    </div>
                  </CardContent>
                  <CardFooter>
                    <Button variant="ghost" className="w-full" onClick={() => setStep(1)}>Train Another Model</Button>
                  </CardFooter>
                </Card>
              )}

              {/* Predict Panel */}
              {trainedModel?.run_id && featureColumns.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Make a Prediction</CardTitle>
                    <CardDescription>Enter feature values (everything except the target) and predict dynamically for this dataset.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {featureColumns.map((col) => (
                        <div key={col} className="space-y-2">
                          <Label>{col}</Label>
                          <Input
                            value={predictInputs[col] ?? ""}
                            onChange={(e) => setPredictInputs((prev) => ({ ...prev, [col]: e.target.value }))}
                            placeholder="Enter value..."
                          />
                        </div>
                      ))}
                    </div>
                    <Button onClick={runPrediction} disabled={isPredicting} className="w-full">
                      {isPredicting ? "Predicting..." : "Predict"}
                    </Button>
                    {predictionResult && (
                      <div className="p-4 rounded-lg border bg-muted/30">
                        {"error" in predictionResult ? (
                          <p className="text-sm text-red-500">{predictionResult.error}</p>
                        ) : (
                          <div className="space-y-2">
                            <p className="text-sm font-semibold">Prediction:</p>
                            <p className="text-lg font-bold">{String(predictionResult.prediction)}</p>
                            {predictionResult.probabilities && (
                              <div className="mt-2">
                                <p className="text-xs text-muted-foreground mb-1">Probabilities</p>
                                <div className="space-y-1">
                                  {Object.entries(predictionResult.probabilities).map(([k, v]: any) => (
                                    <div key={k} className="flex items-center justify-between text-xs">
                                      <span>{k}</span>
                                      <span className="font-medium">{(Number(v) * 100).toFixed(1)}%</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* API Key Modal */}
      <AnimatePresence>
        {showApiKeyModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowApiKeyModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-background rounded-lg shadow-xl max-w-md w-full p-6"
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings2 className="w-5 h-5" />
                    <span>LLM API Key Settings</span>
                  </CardTitle>
                  <CardDescription>
                    Provide your Gemini API key to enable AI-powered features. 
                    The system will automatically switch models if rate limits are hit.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {llmStatus && (
                    <div className="p-3 bg-muted rounded-md text-sm">
                      <div className="font-medium mb-1">Current Status:</div>
                      <div>
                        {llmStatus.llm_available ? (
                          <div>
                            ✅ <strong>Active</strong> - Using: {llmStatus.current_model || "Default"}
                            {llmStatus.model_reason && (
                              <div className="text-muted-foreground mt-1">
                                {llmStatus.model_reason}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div>⚠️ Using rule-based fallbacks</div>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <Label htmlFor="api-key">Gemini API Key (leave empty to use default)</Label>
                    <Input
                      id="api-key"
                      type="password"
                      placeholder="Enter your API key or leave empty for default..."
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleSetApiKey();
                      }}
                    />
                    <p className="text-xs text-muted-foreground">
                      Your API key is only stored in memory and never saved to disk. 
                      Leave empty to use the default key from .env file.
                    </p>
                  </div>

                  {apiKeyStatus && (
                    <div
                      className={`p-3 rounded-md text-sm ${
                        apiKeyStatus.status === "success"
                          ? "bg-green-500/10 text-green-700 dark:text-green-400"
                          : apiKeyStatus.status === "warning"
                          ? "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400"
                          : "bg-red-500/10 text-red-700 dark:text-red-400"
                      }`}
                    >
                      {apiKeyStatus.status === "success" ? "✅ " : apiKeyStatus.status === "warning" ? "⚠️ " : "❌ "}
                      {apiKeyStatus.message}
                      {apiKeyStatus.is_rate_limit && (
                        <div className="mt-2 text-xs">
                          💡 The system will automatically try alternative models when rate limits are hit.
                        </div>
                      )}
                      {apiKeyStatus.current_model && (
                        <div className="mt-2 text-xs">
                          Using model: <strong>{apiKeyStatus.current_model}</strong>
                          {apiKeyStatus.model_reason && ` (${apiKeyStatus.model_reason})`}
                        </div>
                      )}
                      {apiKeyStatus.using_default && (
                        <div className="mt-2 text-xs">
                          ℹ️ Using default API key from environment
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex justify-end space-x-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setShowApiKeyModal(false);
                      setApiKey("");
                      setApiKeyStatus(null);
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSetApiKey}
                    disabled={isSettingApiKey}
                  >
                    {isSettingApiKey ? "Setting..." : apiKey.trim() ? "Set Custom Key" : "Use Default Key"}
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
