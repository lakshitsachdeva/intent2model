"use client";

import { useState, useEffect, Fragment } from "react";
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
  AlertCircle,
  MessageCircle,
  Send,
  Target
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
  // Backend base URLs
  // - Default: `http://localhost:8000`
  // - If you open the frontend via LAN / different hostname, this keeps working.
  const BACKEND_HOST =
    typeof window !== "undefined" ? window.location.hostname : "localhost";
  const BACKEND_HTTP_BASE = `http://${BACKEND_HOST}:8000`;
  const BACKEND_WS_BASE = `ws://${BACKEND_HOST}:8000`;

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
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [isSettingApiKey, setIsSettingApiKey] = useState(false);
  const [selectedLlmProvider, setSelectedLlmProvider] = useState<string>("gemini_cli");
  const [selectedModelName, setSelectedModelName] = useState<string | null>(null);
  const [selectedTaskType, setSelectedTaskType] = useState<"classification" | "regression" | null>(null);
  const [showDevLogs, setShowDevLogs] = useState(false);
  const [devLogs, setDevLogs] = useState<any>(null);
  const [devLogsError, setDevLogsError] = useState<string | null>(null);
  const [backendLogTail, setBackendLogTail] = useState<string[]>([]);
  const [liveLogs, setLiveLogs] = useState<Array<{ts: string, message: string, stage?: string, progress?: number}>>([]);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<string>("");
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const logsEndRef = React.useRef<HTMLDivElement>(null);
  const wsRef = React.useRef<WebSocket | null>(null);
  const wsActiveRef = React.useRef(true); // false when effect cleanup ran — don’t use this socket
  const [wsConnected, setWsConnected] = useState(false);
  const [wsFailed, setWsFailed] = useState(false);
  const [runState, setRunState] = useState<{
    run_id: string;
    status: string;
    current_step: string;
    attempt_count: number;
    progress: number;
    events: Array<{ ts: string; step_name: string; message: string; status?: string; payload?: Record<string, unknown> }>;
  } | null>(null);
  const [logStreamError, setLogStreamError] = useState<string | null>(null);
  const [logStreamStatus, setLogStreamStatus] = useState<"idle" | "connecting" | "streaming" | "error">("idle");
  // User constraints (chat-first: affect next ExecutionPlan)
  const [userConstraints, setUserConstraints] = useState<{
    exclude_models: string[];
    keep_features: string;
    primary_metric: string;
  }>({ exclude_models: [], keep_features: "", primary_metric: "" });
  // Session & chat (chat-first UI)
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<Array<{ role: string; content: string; payload?: Record<string, unknown> }>>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const chatEndRef = React.useRef<HTMLDivElement>(null);
  // Cursor-for-ML: ModelState + live notebook (source of truth for UI)
  const [modelState, setModelState] = useState<{
    dataset_summary?: Record<string, unknown>;
    current_features?: string[];
    preprocessing_steps?: string[];
    current_model?: string;
    previous_model?: string;
    metrics?: Record<string, unknown>;
    error_analysis?: { confusion_matrix?: number[][]; class_labels?: string[]; feature_importance?: Record<string, number> };
    attempt_number?: number;
    last_diff?: Record<string, unknown>;
    status?: string;
    status_message?: string;
  } | null>(null);
  const [notebookCells, setNotebookCells] = useState<Array<{ attempt?: number; model?: string; diff?: Record<string, unknown>; preprocessing?: string[]; primary_metric?: string }>>([]);
  const [sessionExpiredMessage, setSessionExpiredMessage] = useState<string | null>(null);
  // Typing animation for last agent message (dhire dhire)
  const [streamingContent, setStreamingContent] = useState("");
  const [streamingLength, setStreamingLength] = useState(0);
  const streamingDoneRef = React.useRef(true);

  const fetchLlmStatus = async () => {
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/health`);
      const data = await resp.json();
      setLlmStatus(data);
      setBackendOnline(true);
      // IMPORTANT: do NOT override user's selected provider from /health polling.
      // The backend reports its default provider, but user may have chosen a different one for this session.
    } catch (e) {
      // Backend may be restarting / offline — don't spam console.
      setBackendOnline(false);
    }
  };

  // Load persisted provider selection on mount; default to CLI if nothing saved
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("intent2model_llm_provider");
      setSelectedLlmProvider(saved && (saved === "gemini" || saved === "gemini_cli") ? saved : "gemini_cli");
    } catch {
      setSelectedLlmProvider("gemini_cli");
    }
  }, []);

  // Persist provider selection whenever user changes it
  useEffect(() => {
    try {
      window.localStorage.setItem("intent2model_llm_provider", selectedLlmProvider);
    } catch {
      // ignore
    }
  }, [selectedLlmProvider]);

  const handleSetApiKey = async () => {
    // Allow empty to use default
    // if (!apiKey.trim()) {
    //   setApiKeyStatus({ status: "error", message: "Please enter an API key" });
    //   return;
    // }

    setIsSettingApiKey(true);
    setApiKeyStatus(null);

    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/api/set-api-key`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, provider: selectedLlmProvider }),
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

  // Fetch LLM status on mount and auto-refresh every 5 seconds (to detect .env changes)
  useEffect(() => {
    fetchLlmStatus();
    // Auto-refresh LLM status every 5 seconds to detect .env changes
    const interval = setInterval(() => {
      fetchLlmStatus();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    setIsUploading(true);
    const file = acceptedFiles[0];
    setFiles([file]);

    try {
      setSessionExpiredMessage(null);
      const uploadResult = await uploadDataset(file);
      setDatasetId(uploadResult.dataset_id);
      setSessionId(uploadResult.session_id || null);
      setChatHistory(
        Array.isArray(uploadResult.chat_history)
          ? uploadResult.chat_history
          : uploadResult.initial_message
            ? [uploadResult.initial_message]
            : []
      );
      const numeric = uploadResult.profile?.numeric_cols || [];
      const categorical = uploadResult.profile?.categorical_cols || [];
      setAvailableColumns([...numeric, ...categorical]);
      setDatasetSummary(null);
      // Fetch visualization summary
      try {
        const s = await fetch(`${BACKEND_HTTP_BASE}/dataset/${uploadResult.dataset_id}/summary`);
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
        alert(`⚠️ Cannot connect to backend. Make sure backend is running on ${BACKEND_HTTP_BASE}\n\nRun: cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`);
      }
      
      // Don't proceed; training needs a dataset_id
      setStep(1);
    } finally {
      setIsUploading(false);
    }
  };

  const uploadDataset = async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BACKEND_HTTP_BASE}/upload`, {
      method: "POST",
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary for FormData
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail || "Dataset upload failed");
    }
    if (!data?.dataset_id) {
      throw new Error("Upload succeeded but backend did not return dataset_id");
    }
    return data;
  };

  const clearSessionExpired = () => {
    setSessionId(null);
    setChatHistory([]);
    setModelState(null);
    setNotebookCells([]);
    setSessionExpiredMessage("Session expired (e.g. backend restarted). Please upload your dataset again.");
  };

  const sendChatMessage = async () => {
    const msg = (chatInput || "").trim();
    if (!msg || !sessionId || chatSending) return;
    setChatInput("");
    setSessionExpiredMessage(null);
    // Optimistic: show user message and "Thinking..." immediately
    setChatHistory((prev) => [
      ...prev,
      { role: "user", content: msg },
      { role: "agent", content: "Thinking…", isPlaceholder: true },
    ]);
    setChatSending(true);
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/session/${sessionId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      if (resp.status === 404) {
        clearSessionExpired();
        setChatSending(false);
        return;
      }
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail || "Chat failed");
      }
      const data = await resp.json();
      if (Array.isArray(data.chat_history)) {
        setChatHistory(data.chat_history);
      }
      // Agent mediates: backend can signal "trigger_training" so we start training (never silent retry)
      const lower = msg.toLowerCase().trim();
      const trainPhrases = [
        "yes", "train", "start training", "go", "run", "start",
        "lets do it", "let's do it", "lets go", "let's go", "do it", "go ahead", "run it", "execute", "run the plan",
        "sure", "ok", "okay", "alright", "chalo", "jao", "karo", "chalo karo",
      ];
      const isTrainCommand =
        trainPhrases.includes(lower) ||
        (lower.startsWith("train") && lower.length <= 35);
      const triggerFromBackend = (data as { trigger_training?: boolean }).trigger_training === true;
      const lastReply = Array.isArray(data.chat_history) && data.chat_history.length > 0
        ? data.chat_history[data.chat_history.length - 1]
        : null;
      const agentSaidStartTraining =
        lastReply?.role === "agent" &&
        typeof lastReply?.content === "string" &&
        lastReply.content.trim() === "Starting training.";
      if (isTrainCommand || triggerFromBackend || agentSaidStartTraining) {
        startSessionTraining();
      }
    } catch (e: unknown) {
      const err = e instanceof Error ? e.message : "Chat failed";
      setChatHistory((prev) => {
        const withoutPlaceholder = prev.filter((m) => !(m as { isPlaceholder?: boolean }).isPlaceholder);
        return [
          ...withoutPlaceholder,
          { role: "agent", content: `Error: ${err}` },
        ];
      });
    } finally {
      setChatSending(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // When last message is agent, start typing animation
  useEffect(() => {
    const last = chatHistory[chatHistory.length - 1];
    if (last?.role === "agent" && typeof last.content === "string") {
      setStreamingContent(last.content);
      setStreamingLength(0);
      streamingDoneRef.current = false;
    } else {
      streamingDoneRef.current = true;
    }
  }, [chatHistory]);

  // Animate streaming length (dhire dhire)
  useEffect(() => {
    if (streamingContent.length === 0 || streamingLength >= streamingContent.length) {
      streamingDoneRef.current = true;
      return;
    }
    const t = setTimeout(() => {
      setStreamingLength((prev) => {
        if (prev >= streamingContent.length) return prev;
        return prev + 1;
      });
    }, 18);
    return () => clearTimeout(t);
  }, [streamingContent, streamingLength]);

  const refreshSession = async () => {
    if (!sessionId) return;
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/session/${sessionId}`);
      if (resp.status === 404) {
        clearSessionExpired();
        return;
      }
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data.chat_history)) setChatHistory(data.chat_history);
      if (data.model_state) setModelState(data.model_state);
      if (Array.isArray(data.notebook_cells)) setNotebookCells(data.notebook_cells);
      if (data.current_run_id) setCurrentRunId(data.current_run_id);
    } catch (e) {
      console.error("Failed to refresh session:", e);
    }
  };

  const startSessionTraining = async () => {
    if (!sessionId || training) return;
    setTraining(true);
    setTrainingError(null);
    setProgress(0);
    setLiveLogs([]);
    setCurrentRunId(null);
    setSessionExpiredMessage(null);
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/session/${sessionId}/train`, { method: "POST" });
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 404) {
        clearSessionExpired();
        return;
      }
      if (data.run_id) {
        setCurrentRunId(data.run_id);
        await fetchRunState(data.run_id);
      }
      await refreshSession();
      if (data.refused) {
        setTrainingError(data.refusal_reason || "Training refused.");
      } else {
        setTrainedModel({ ...data, run_id: data.run_id });
        setSelectedModelName(data.metrics?.model_name || data.model_state?.current_model);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Training failed";
      setTrainingError(msg);
    } finally {
      setTraining(false);
      setProgress(100);
    }
  };

  const cancelSessionTraining = async () => {
    if (!sessionId) return;
    try {
      await fetch(`${BACKEND_HTTP_BASE}/session/${sessionId}/cancel`, { method: "POST" });
      await refreshSession();
    } catch (e) {
      console.error("Cancel failed:", e);
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
    setLiveLogs([]); // Clear previous logs
    setCurrentStage("");
    setCurrentRunId(null); // Will be set when we get run_id
    setTrainingError(null);
    
    // Extract target column from intent or use first available
    let targetColumn = intent.trim();
    if (!targetColumn && availableColumns.length > 0) {
      targetColumn = availableColumns[0];
    }
    
    // Progress tracking
    let progressInterval: NodeJS.Timeout | null = null;
    let logsInterval: NodeJS.Timeout | null = null;
    
    try {
      // If we don't have datasetId (common after restart), auto-upload the last selected file
      let ensuredDatasetId = datasetId;
      if (!ensuredDatasetId) {
        const file = files?.[0];
        if (!file) {
          throw new Error("No dataset available. Please upload a CSV file first.");
        }
        setIsUploading(true);
        const uploadResult = await uploadDataset(file);
        ensuredDatasetId = uploadResult.dataset_id;
        setDatasetId(ensuredDatasetId);
        const numeric = uploadResult.profile?.numeric_cols || [];
        const categorical = uploadResult.profile?.categorical_cols || [];
        setAvailableColumns([...numeric, ...categorical]);
        setIsUploading(false);
      }

      // Start progress simulation (will be overridden by real progress from logs)
      progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) {
            return 95; // Don't go to 100 until training completes
          }
          return prev + Math.random() * 2;
        });
      }, 1000);
      
      // Start polling backend logs immediately (before we get run_id) - faster!
      logsInterval = setInterval(() => {
        fetchBackendLogTail();
      }, 200);

      const body: Record<string, unknown> = {
        dataset_id: ensuredDatasetId,
        target: targetColumn,
        task: selectedTaskType || undefined,
        llm_provider: selectedLlmProvider,
      };
      if (userConstraints.exclude_models.length > 0 || userConstraints.keep_features.trim() || userConstraints.primary_metric.trim()) {
        body.user_constraints = {
          ...(userConstraints.exclude_models.length ? { exclude_models: userConstraints.exclude_models } : {}),
          ...(userConstraints.keep_features.trim() ? { keep_features: userConstraints.keep_features.split(",").map((s) => s.trim()).filter(Boolean) } : {}),
          ...(userConstraints.primary_metric.trim() ? { primary_metric: userConstraints.primary_metric.trim() } : {}),
        };
      }
      const response = await fetch(`${BACKEND_HTTP_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await response.json();
      
      if (response.ok) {
        // Check if training was REFUSED (epistemically honest failure)
        if (data.status === "refused" || data.refused) {
          if (progressInterval) clearInterval(progressInterval);
          if (logsInterval) clearInterval(logsInterval);
          setProgress(100);
          setTraining(false);
          setTrainingError(
            `🛑 Training REFUSED: ${data.refusal_reason || "Model quality unacceptable"}\n\n` +
            `Failed gates:\n${(data.failed_gates || []).map((g: string) => `  - ${g}`).join("\n")}\n\n` +
            `This is an epistemically honest refusal - the model would not produce usable predictions.`
          );
          // Still show the refusal data so user can see what happened
          setTrainedModel({
            ...data,
            refused: true,
            status: "refused"
          });
          return;
        }
        
        // Set run_id for real-time log polling (run_id is created at start of training)
        if (data.run_id) {
          setCurrentRunId(data.run_id);
          // Start polling logs immediately
          fetchRunLogs(data.run_id);
        }
        
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
      } else if (data.run_id) {
        // Training might still be in progress - we got run_id, start polling run logs
        setCurrentRunId(data.run_id);
        fetchRunLogs(data.run_id);
        // Keep training state true and continue polling
      } else {
        // Try with first available column if specified column fails
        if (availableColumns.length > 0 && targetColumn !== availableColumns[0]) {
          const fallbackResponse = await fetch(`${BACKEND_HTTP_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset_id: ensuredDatasetId,
              target: availableColumns[0],
              task: selectedTaskType || undefined,
              llm_provider: selectedLlmProvider,
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
      setIsUploading(false);
      if (progressInterval) clearInterval(progressInterval);
      if (logsInterval) clearInterval(logsInterval);
      // Show the error and KEEP the training screen so logs remain visible
      const msg =
        (error as any)?.message ||
        "Training failed. Check Developer Logs / Live Activity for details.";
      setTrainingError(msg);
      setTraining(false);
    }
  };

  const fetchRunLogs = async (runId: string) => {
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/run/${runId}/logs?limit=200`);
      if (!resp.ok) {
        const err = await resp.text();
        setDevLogsError(err || "Failed to fetch logs");
        return;
      }
      const data = await resp.json();
      setDevLogs(data);
      setDevLogsError(null);
      
      // Update live logs for real-time display
      if (data.events && Array.isArray(data.events)) {
        setLiveLogs(data.events);
      }
      
      // If backend provides progress, prefer it
      if (typeof data.progress === "number") {
        setProgress(Math.max(0, Math.min(100, data.progress)));
      }
      
      // Update current stage
      if (data.stage) {
        setCurrentStage(data.stage);
      }
    } catch (e: any) {
      setDevLogsError(e?.message || "Failed to fetch logs");
    }
  };

  const fetchRunState = async (runId: string) => {
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/runs/${runId}`);
      if (!resp.ok) {
        setLogStreamError((prev) => prev || `Run logs returned ${resp.status}. Retrying…`);
        return;
      }
      const data = await resp.json();
      setRunState(data);
      setLogStreamError(null);
      if ((data?.events?.length ?? 0) > 0) setLogStreamStatus("streaming");
    } catch (e) {
      setLogStreamError((prev) => prev || "Run logs unreachable. Retrying…");
    }
  };
  
  // Poll logs in real-time during training — fast polling so logs feel real-time
  useEffect(() => {
    if (!training) {
      setLogStreamStatus("idle");
      setLogStreamError(null);
      return;
    }
    setLogStreamStatus("connecting");
    setLogStreamError(null);

    const POLL_RUN_MS = 150;   // run state every 150ms for real-time feel
    const POLL_BACKEND_MS = 200;
    const POLL_LATEST_ID_MS = 200;

    const fetchLatestRunId = async () => {
      try {
        const r = await fetch(`${BACKEND_HTTP_BASE}/run/latest-id`);
        const d = await r.json();
        if (d?.run_id) {
          setCurrentRunId((prev) => (prev ? prev : d.run_id));
          setLogStreamError(null);
        }
      } catch (e) {
        setLogStreamError((prev) => prev || "Could not get run ID. Retrying…");
      }
    };
    fetchLatestRunId();
    const latestIdInterval = setInterval(fetchLatestRunId, POLL_LATEST_ID_MS);

    const backendInterval = setInterval(() => fetchBackendLogTail(), POLL_BACKEND_MS);

    let runLogInterval: NodeJS.Timeout | null = null;
    if (currentRunId) {
      fetchRunLogs(currentRunId);
      fetchRunState(currentRunId);
      runLogInterval = setInterval(() => {
        fetchRunLogs(currentRunId);
        fetchRunState(currentRunId);
      }, POLL_RUN_MS);
    }

    return () => {
      clearInterval(latestIdInterval);
      clearInterval(backendInterval);
      if (runLogInterval) clearInterval(runLogInterval);
    };
  }, [training, currentRunId]);

  // Fetch run state on results step so "What happened?" timeline is available after completion
  useEffect(() => {
    const runId = trainedModel?.run_id || currentRunId;
    if (runId && step === 4) fetchRunState(runId);
  }, [step, trainedModel?.run_id, currentRunId]);
  
  // Auto-scroll logs to bottom when new logs arrive
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [liveLogs, backendLogTail, runState?.events?.length]);

  const fetchBackendLogTail = async () => {
    try {
      const resp = await fetch(`${BACKEND_HTTP_BASE}/logs/backend?limit=500`);
      const data = await resp.json();
      const lines = Array.isArray(data.lines) ? data.lines : [];
      setBackendLogTail(lines);
      setBackendOnline(true);
      if (training && lines.length > 0) setLogStreamStatus("streaming");
      setLogStreamError(null);

      if (!currentRunId && training && lines.length) {
        const joined = lines.slice(-80).join("\n");
        const m = joined.match(/Run ID created:\s*([0-9a-fA-F-]{36})/);
        if (m && m[1]) {
          setCurrentRunId(m[1]);
          fetchRunLogs(m[1]);
        }
      }
    } catch (e) {
      setBackendOnline(false);
      setLogStreamError((prev) => prev || "Backend logs unreachable. Retrying…");
    }
  };

  // WebSocket connection for real-time logs
  useEffect(() => {
    if (!training && !showDevLogs) {
      wsActiveRef.current = false;
      if (wsRef.current) {
        if (wsRef.current.readyState === WebSocket.OPEN) wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    wsActiveRef.current = true;
    const ws = new WebSocket(`${BACKEND_WS_BASE}/ws/logs`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!wsActiveRef.current) {
        ws.close();
        return;
      }
      setWsConnected(true);
      setWsFailed(false);
      setBackendOnline(true);
    };

    ws.onmessage = (event) => {
      if (!wsActiveRef.current) return;
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === "connected" || data.type === "pong") {
          return;
        }

        if (data.message && data.run_id) {
          // Extract run_id if we don't have it yet
          setCurrentRunId((prevRunId) => {
            if (!prevRunId && data.run_id) {
              return data.run_id;
            }
            return prevRunId;
          });

          // Add to live logs
          setLiveLogs((prev) => {
            const updated = [...prev, data];
            // Keep last 1000 events
            return updated.slice(-1000);
          });

          // Update progress and stage
          if (typeof data.progress === "number") {
            setProgress(Math.max(0, Math.min(100, data.progress)));
          }
          if (data.stage) {
            setCurrentStage(data.stage);
          }
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onerror = (error) => {
      // Browser WS errors are intentionally opaque ("{}"). Don't spam console and don't mark backend offline
      // because HTTP may still work. We fall back to HTTP polling in that case.
      setWsConnected(false);
      setWsFailed(true);
    };

    ws.onclose = () => {
      wsRef.current = null;
      setWsConnected(false);
      // Try to reconnect after 2 seconds if still training
      const shouldReconnect = training || showDevLogs;
      if (shouldReconnect) {
        setTimeout(() => {
          // Check again if we should reconnect
          if (training || showDevLogs) {
            // Reconnect handled by effect re-run (training/showDevLogs still true)
            setWsFailed(true);
          }
        }, 2000);
      }
    };

    // Send ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("ping");
      }
    }, 30000);

    return () => {
      wsActiveRef.current = false;
      clearInterval(pingInterval);
      // Only close if already open — closing while CONNECTING causes "closed before connection" error
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      wsRef.current = null;
    };
  }, [training, showDevLogs]);

  // Fallback: poll backend logs if WS is failing or dev panel is open (WS is primary).
  useEffect(() => {
    const shouldPoll = training && (showDevLogs || wsFailed);
    if (!shouldPoll) return;
    fetchBackendLogTail();
    const t = setInterval(() => fetchBackendLogTail(), wsFailed ? 800 : 5000);
    return () => clearInterval(t);
  }, [training, showDevLogs, wsFailed]);

  const selectModel = async (modelName: string) => {
    if (!trainedModel?.run_id) return;
    try {
      // Optimistic UI
      setSelectedModelName(modelName);
      setTrainedModel((prev: any) => ({ ...(prev || {}), selected_model: modelName }));
      await fetch(`${BACKEND_HTTP_BASE}/run/select-model`, {
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
      const resp = await fetch(`${BACKEND_HTTP_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: trainedModel.run_id, features }),
      });
      const out = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const msg = (out as { detail?: string }).detail ?? (out as { message?: string }).message ?? resp.statusText ?? "Prediction failed";
        setPredictionResult({ error: typeof msg === "string" ? msg : JSON.stringify(msg) });
        return;
      }
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
          Pair-program with an ML engineer on your dataset. Chat, steer, and see the model evolve live.
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

      {/* Cursor-for-ML: no wizard stepper — Upload OR Chat + Live panels */}
      {!sessionId && (
        <div className="space-y-4 pb-4 border-b-2 border-muted">
          {sessionExpiredMessage && (
            <div className="rounded-lg border border-amber-500/50 bg-amber-50 dark:bg-amber-950/30 px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
              {sessionExpiredMessage}
            </div>
          )}
          <div className="grid grid-cols-1 gap-4">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Database className="w-5 h-5" />
              <span className="text-sm font-medium">Upload a dataset to start</span>
            </div>
          </div>
        </div>
      )}

      {/* Live ML Chat + Live panels — ALWAYS visible after upload (single view) */}
      {sessionId && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Chat — persistent, always visible */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <MessageCircle className="w-4 h-4" />
                  Chat — pair-program with the ML engineer
                </CardTitle>
                <CardDescription>
                  Say &quot;drop id&quot;, &quot;use species as target&quot;, &quot;yes&quot; or &quot;try something stronger&quot;. I never retry silently.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="rounded-lg border bg-muted/30 max-h-[320px] overflow-y-auto p-3 space-y-3">
                  {chatHistory.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No messages yet. Say &quot;use X as target&quot; or &quot;yes&quot; to train.</p>
                  ) : (
                chatHistory.map((m, idx) => {
                  const isPlaceholder = (m as { isPlaceholder?: boolean }).isPlaceholder;
                  const isLastAgent = idx === chatHistory.length - 1 && m.role === "agent";
                  const showStreaming = !isPlaceholder && isLastAgent && streamingContent === m.content && streamingLength < (m.content?.length ?? 0);
                  const displayContent = showStreaming
                    ? (m.content as string).slice(0, streamingLength) + "▌"
                    : (m.content as string);
                  // Render **bold** as actual bold (Markdown-style)
                  const boldSegments = (displayContent as string).split(/\*\*/);
                  return (
                    <div
                      key={idx}
                      className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                          m.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted border"
                        } ${isPlaceholder ? "animate-pulse" : ""}`}
                      >
                        <p className="whitespace-pre-wrap">
                          {boldSegments.map((seg, i) =>
                            i % 2 === 1 ? <strong key={i}>{seg}</strong> : <Fragment key={i}>{seg}</Fragment>
                          )}
                        </p>
                      </div>
                    </div>
                  );
                })
                  )}
                  <div ref={chatEndRef} />
                </div>
                <div className="flex flex-wrap gap-2">
                  <Input
                    placeholder="e.g. use species as target, or say 'train' / 'yes' to start training"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendChatMessage()}
                    className="flex-1 min-w-[200px]"
                    disabled={chatSending || training}
                  />
                  <Button onClick={sendChatMessage} disabled={chatSending || !chatInput.trim() || training} size="sm">
                    <Send className="w-4 h-4 mr-1" /> Send
                  </Button>
                  <Button onClick={cancelSessionTraining} disabled={!training} size="sm" variant="outline">
                    Cancel
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">Say &quot;train&quot; or &quot;yes&quot; to start training — no separate button.</p>
              </CardContent>
            </Card>
          </div>

          {/* Right: Live panels — Model state, Notebook, Visuals */}
          <div className="space-y-4">
            {/* Live Activity — backend.log style during training */}
            {(training || (runState?.events?.length ?? 0) > 0) && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Zap className="w-4 h-4" /> Live Activity
                  </CardTitle>
                  <CardDescription className="text-xs">Training log (like backend.log) — updates in real time</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-lg border-2 border-emerald-500/30 bg-slate-900 text-green-400 font-mono text-xs p-3 max-h-[240px] overflow-y-auto">
                    {runState?.events?.length ? (
                      runState.events.slice(-60).map((ev: { ts?: string; step_name?: string; message?: string; status?: string }, idx: number) => {
                        const runShort = (currentRunId || runState?.run_id || "").slice(0, 8);
                        const isFailed = ev.status === "failed" || (ev.message && (ev.message.includes("❌") || ev.message.includes("failed")));
                        const isSuccess = ev.message && (ev.message.includes("✅") || ev.message.includes("succeeded"));
                        const lineCl = isFailed ? "text-red-400" : isSuccess ? "text-emerald-300" : "text-green-400";
                        return (
                          <div key={idx} className={`${lineCl} whitespace-pre-wrap`}>
                            <span className="text-slate-500 select-none">[{runShort}]</span>{" "}
                            <span className="text-slate-400">[{ev.step_name || "info"}]</span>{" "}
                            {ev.message}
                          </div>
                        );
                      })
                    ) : training ? (
                      <div className="text-slate-500">Connecting… logs will appear here.</div>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Model state (source of truth) */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" /> Model state
                </CardTitle>
                <CardDescription className="text-xs">Dataset, features, preprocessing, model, metrics — live</CardDescription>
              </CardHeader>
              <CardContent className="text-xs space-y-2">
                {modelState ? (
                  <>
                    {modelState.dataset_summary && (
                      <p>Rows: {String((modelState.dataset_summary as Record<string, unknown>).n_rows ?? "—")} × Cols: {String((modelState.dataset_summary as Record<string, unknown>).n_cols ?? "—")}</p>
                    )}
                    {modelState.current_model && <p><strong>Model:</strong> {modelState.current_model}</p>}
                    {modelState.preprocessing_steps?.length ? <p><strong>Preprocessing:</strong> {modelState.preprocessing_steps.slice(0, 5).join(", ")}</p> : null}
                    {modelState.current_features?.length ? <p><strong>Features:</strong> {modelState.current_features.length} used</p> : null}
                    {modelState.attempt_number ? <p><strong>Attempt:</strong> {modelState.attempt_number}</p> : null}
                    {Object.keys(modelState.last_diff || {}).length > 0 && (
                      <p className="text-primary"><strong>Last diff:</strong> {JSON.stringify(modelState.last_diff)}</p>
                    )}
                    {modelState.metrics && Object.keys(modelState.metrics).length > 0 && (
                      <p><strong>Metrics:</strong> {Object.entries(modelState.metrics).filter(([k]) => !k.startsWith("_")).slice(0, 5).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`).join(", ")}</p>
                    )}
                  </>
                ) : (
                  <p className="text-muted-foreground">Run training to see state.</p>
                )}
              </CardContent>
            </Card>

            {/* Live notebook / code view (diff-based) */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <FileJson className="w-4 h-4" /> Live notebook
                </CardTitle>
                <CardDescription className="text-xs">Each attempt = one cell (model, diff, preprocessing)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="max-h-[200px] overflow-y-auto space-y-2 text-xs">
                  {notebookCells.length === 0 ? (
                    <p className="text-muted-foreground">No attempts yet. Say &quot;train&quot; to run.</p>
                  ) : (
                    notebookCells.slice(-10).map((cell, idx) => (
                      <div key={idx} className="rounded border p-2 bg-muted/30">
                        <span className="font-medium">Attempt {cell.attempt ?? idx + 1}</span>: {cell.model ?? "—"} → {cell.primary_metric ?? "—"}
                        {cell.diff && Object.keys(cell.diff).length > 0 && <div className="text-primary mt-1">Diff: {JSON.stringify(cell.diff)}</div>}
                      </div>
                    ))
                  )}
                </div>
                {(currentRunId || sessionId) && (
                  <div className="flex flex-wrap gap-2 pt-2 border-t">
                    {currentRunId && (
                      <a
                        href={`${BACKEND_HTTP_BASE}/download/${currentRunId}/notebook`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                      >
                        <FileJson className="w-3 h-3" /> Download .ipynb
                      </a>
                    )}
                    {sessionId && (
                      <a
                        href={`${BACKEND_HTTP_BASE}/session/${sessionId}/notebook`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                      >
                        View notebook (session)
                      </a>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Artifacts & downloads — notebook, report, model, charts (all) */}
            {currentRunId && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4" /> Artifacts &amp; downloads
                  </CardTitle>
                  <CardDescription className="text-xs">Notebook, report, model, and charts for this run</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-wrap gap-2">
                  <a
                    href={`${BACKEND_HTTP_BASE}/download/${currentRunId}/notebook`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                  >
                    <FileJson className="w-3 h-3" /> Notebook (.ipynb)
                  </a>
                  <a
                    href={`${BACKEND_HTTP_BASE}/download/${currentRunId}/report`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                  >
                    Report
                  </a>
                  <a
                    href={`${BACKEND_HTTP_BASE}/download/${currentRunId}/model`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                  >
                    Model
                  </a>
                  <a
                    href={`${BACKEND_HTTP_BASE}/download/${currentRunId}/readme`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                  >
                    Readme
                  </a>
                  <a
                    href={`${BACKEND_HTTP_BASE}/download/${currentRunId}/all`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary hover:underline inline-flex items-center gap-1"
                  >
                    All (zip: notebook + report + charts)
                  </a>
                </CardContent>
              </Card>
            )}

            {/* Charts — metrics & feature importance from modelState */}
            {(modelState?.metrics && Object.keys(modelState.metrics).length > 0) || (modelState?.error_analysis?.feature_importance && Object.keys(modelState.error_analysis.feature_importance).length > 0) ? (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" /> Charts
                  </CardTitle>
                  <CardDescription className="text-xs">Metrics and feature importance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {modelState?.metrics && Object.keys(modelState.metrics).length > 0 && (() => {
                    const m = modelState.metrics as Record<string, unknown>;
                    const skip = (k: string) => k.startsWith("_") || typeof m[k] !== "number";
                    const chartData = Object.entries(m)
                      .filter(([k]) => !skip(k))
                      .slice(0, 10)
                      .map(([name, value]) => ({ name, value: Number(value) }));
                    if (chartData.length === 0) return null;
                    return (
                      <div className="h-44">
                        <p className="font-medium mb-1 text-xs">Metrics</p>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 25 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" angle={-25} textAnchor="end" height={50} tick={{ fontSize: 10 }} />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip />
                            <Bar dataKey="value" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    );
                  })()}
                  {modelState?.error_analysis?.feature_importance && Object.keys(modelState.error_analysis.feature_importance).length > 0 && (
                    <div className="h-44">
                      <p className="font-medium mb-1 text-xs">Feature importance</p>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={Object.entries(modelState.error_analysis.feature_importance as Record<string, number>)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 8)
                            .map(([name, value]) => ({ name: name.length > 12 ? name.slice(0, 12) + "…" : name, value }))}
                          layout="vertical"
                          margin={{ top: 5, right: 5, left: 60, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" tick={{ fontSize: 10 }} />
                          <YAxis type="category" dataKey="name" width={58} tick={{ fontSize: 9 }} />
                          <Tooltip />
                          <Bar dataKey="value" fill="hsl(var(--chart-2))" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
            ) : null}

            {/* Predict — enter feature values and get a prediction (visible place to predict) */}
            {currentRunId && featureColumns.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Target className="w-4 h-4" /> Predict
                  </CardTitle>
                  <CardDescription className="text-xs">Enter feature values and get a prediction from the trained model</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-1 gap-2">
                    {featureColumns.map((col) => (
                      <div key={col} className="space-y-1">
                        <Label className="text-xs">{col}</Label>
                        <Input
                          className="h-8 text-xs"
                          value={predictInputs[col] ?? ""}
                          onChange={(e) => setPredictInputs((prev) => ({ ...prev, [col]: e.target.value }))}
                          placeholder="value..."
                        />
                      </div>
                    ))}
                  </div>
                  <Button size="sm" onClick={runPrediction} disabled={isPredicting} className="w-full">
                    {isPredicting ? "Predicting…" : "Predict"}
                  </Button>
                  {predictionResult && (
                    <div className="p-2 rounded border bg-muted/30 text-xs">
                      {"error" in predictionResult ? (
                        <p className="text-red-500">{predictionResult.error}</p>
                      ) : (
                        <div className="space-y-1">
                          <p className="font-semibold">Result: {predictionResult.prediction !== undefined && predictionResult.prediction !== null ? String(predictionResult.prediction) : "—"}</p>
                          {predictionResult.probabilities && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {Object.entries(predictionResult.probabilities).map(([k, v]: [string, unknown]) => (
                                <span key={k} className="bg-background px-1 rounded">{k}: {(Number(v) * 100).toFixed(0)}%</span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Real-time visuals: confusion matrix, feature importance */}
            {(modelState?.error_analysis?.confusion_matrix?.length || (modelState?.error_analysis?.feature_importance && Object.keys(modelState.error_analysis.feature_importance).length)) ? (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Error analysis</CardTitle>
                  <CardDescription className="text-xs">Confusion matrix & feature importance</CardDescription>
                </CardHeader>
                <CardContent className="text-xs space-y-3">
                  {modelState?.error_analysis?.confusion_matrix?.length ? (
                    <div>
                      <p className="font-medium mb-1">Confusion matrix</p>
                      <div className="overflow-x-auto">
                        <table className="border border-collapse">
                          <tbody>
                            {(modelState.error_analysis.confusion_matrix as number[][]).slice(0, 8).map((row, i) => (
                              <tr key={i}>
                                {row.slice(0, 8).map((v, j) => (
                                  <td key={j} className="border px-1 text-right">{v}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {modelState.error_analysis.class_labels?.length ? (
                          <p className="mt-1 text-muted-foreground">Classes: {modelState.error_analysis.class_labels.join(", ")}</p>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                  {modelState?.error_analysis?.feature_importance && Object.keys(modelState.error_analysis.feature_importance).length > 0 ? (
                    <div>
                      <p className="font-medium mb-1">Feature importance (top 8)</p>
                      <ul className="list-disc pl-4">
                        {Object.entries(modelState.error_analysis.feature_importance)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 8)
                          .map(([name, val]) => (
                            <li key={name}>{name}: {(val as number).toFixed(4)}</li>
                          ))}
                      </ul>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            ) : null}
          </div>
        </div>
      )}

      <AnimatePresence mode="wait">
        {step === 1 && !sessionId && (
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

        {step === 2 && !sessionId && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            key="step2"
          >
            <Card>
              <CardHeader>
                <CardTitle>Define your Model&apos;s Intent</CardTitle>
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
                  <div
                    className={`p-4 rounded-lg border bg-card cursor-pointer transition-colors ${
                      selectedTaskType === "regression" ? "border-primary ring-2 ring-primary/20" : "hover:border-primary/50"
                    }`}
                    onClick={() => setSelectedTaskType("regression")}
                  >
                    <h4 className="font-semibold mb-1">Regression</h4>
                    <p className="text-xs text-muted-foreground">Predict continuous values (prices, time, quantity)</p>
                  </div>
                  <div
                    className={`p-4 rounded-lg border bg-card cursor-pointer transition-colors ${
                      selectedTaskType === "classification" ? "border-primary ring-2 ring-primary/20" : "hover:border-primary/50"
                    }`}
                    onClick={() => setSelectedTaskType("classification")}
                  >
                    <h4 className="font-semibold mb-1">Classification</h4>
                    <p className="text-xs text-muted-foreground">Categorize data into discrete labels</p>
                  </div>
                </div>

                <Separator />
                <div className="space-y-3">
                  <Label className="text-sm font-medium">User constraints (override LLM)</Label>
                  <p className="text-xs text-muted-foreground">These affect the next run: exclude models, keep features, or choose metric.</p>
                  <div className="flex flex-wrap gap-2">
                    {["random_forest", "xgboost", "svm", "gradient_boosting", "naive_bayes"].map((m) => (
                      <label key={m} className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={userConstraints.exclude_models.includes(m)}
                          onChange={() => {
                            setUserConstraints((prev) => ({
                              ...prev,
                              exclude_models: prev.exclude_models.includes(m)
                                ? prev.exclude_models.filter((x) => x !== m)
                                : [...prev.exclude_models, m],
                            }));
                          }}
                          className="rounded"
                        />
                        <span className="text-sm">{m.replace(/_/g, " ")}</span>
                      </label>
                    ))}
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs">Keep features (comma-separated)</Label>
                      <Input
                        placeholder="e.g. col1, col2"
                        value={userConstraints.keep_features}
                        onChange={(e) => setUserConstraints((prev) => ({ ...prev, keep_features: e.target.value }))}
                        className="mt-1 h-9"
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Primary metric override</Label>
                      <select
                        className="w-full mt-1 h-9 rounded-md border border-input bg-background px-3 text-sm"
                        value={userConstraints.primary_metric}
                        onChange={(e) => setUserConstraints((prev) => ({ ...prev, primary_metric: e.target.value }))}
                      >
                        <option value="">Default</option>
                        {selectedTaskType === "regression" ? (
                          <>
                            <option value="rmse">RMSE</option>
                            <option value="mae">MAE</option>
                            <option value="r2">R²</option>
                          </>
                        ) : (
                          <>
                            <option value="accuracy">Accuracy</option>
                            <option value="f1">F1</option>
                            <option value="recall">Recall</option>
                            <option value="roc_auc">ROC AUC</option>
                          </>
                        )}
                      </select>
                    </div>
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

        {step === 3 && !sessionId && (
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
                    {trainingError && (
                      <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-300">
                        <div className="font-semibold mb-1">Training failed</div>
                        <div className="whitespace-pre-wrap wrap-break-word">{trainingError}</div>
                        <div className="mt-2 text-xs text-muted-foreground">
                          Keep this screen open — logs will continue updating below.
                        </div>
                      </div>
                    )}
                    {/* Progress Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm font-medium">
                        <span>{currentStage ? `Stage: ${currentStage}` : "Training in progress"}</span>
                        <span>{Math.round(progress)}%</span>
                      </div>
                      <Progress value={progress} className="h-3" />
                    </div>
                    
                    {/* Old Progress Indicators - Keep these! */}
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
                    
                    {/* Live Logs — real-time streaming with error handling */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between flex-wrap gap-2">
                        <h4 className="text-sm font-semibold">Live Activity — real-time</h4>
                        <div className="flex items-center gap-2">
                          {logStreamError && (
                            <Badge variant="destructive" className="text-xs">
                              {logStreamError}
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs font-mono">
                            {(runState?.events?.length ?? 0) || liveLogs.length} events
                          </Badge>
                        </div>
                      </div>
                      {logStreamError && (
                        <div className="rounded-md border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-300">
                          Log stream had errors. Polling will retry. If backend is down, start it with <code className="text-xs bg-black/20 px-1 rounded">./backend/run.sh</code>
                        </div>
                      )}
                      <div
                        className="rounded-lg border-2 border-emerald-500/30 bg-slate-900 text-green-400 font-mono text-xs p-3 max-h-[320px] overflow-y-auto overflow-x-auto"
                        ref={logsEndRef}
                      >
                        {(training && (runState?.events?.length ?? 0) > 0) || (training && backendLogTail.length > 0) || liveLogs.length > 0 ? (
                          <div className="space-y-0.5">
                            {/* Primary: run state timeline (polled every 150ms for real-time) */}
                            {runState?.events?.length ? runState.events.slice(-80).map((ev: any, idx: number) => {
                              const isFailed = ev.status === "failed" || (ev.message && (ev.message.includes("❌") || ev.message.includes("failed") || ev.message.includes("REFUSED")));
                              const isSuccess = ev.message && (ev.message.includes("✅") || ev.message.includes("succeeded"));
                              const isWarn = ev.message && (ev.message.includes("⚠️") || ev.message.includes("WARNING"));
                              const isRepair = (ev.step_name || ev.stage || "").toLowerCase().includes("repair") || (ev.message && ev.message.includes("🔧"));
                              const isRetry = (ev.step_name || ev.stage || "").toLowerCase().includes("retry") || (ev.message && ev.message.includes("🔄"));
                              const isDiagnose = (ev.step_name || ev.stage || "").toLowerCase().includes("diagnose");
                              const lineCl = isFailed ? "text-red-400" : isSuccess ? "text-emerald-300" : isWarn ? "text-amber-400" : isRepair ? "text-cyan-400" : isRetry ? "text-violet-400" : isDiagnose ? "text-blue-300" : "text-green-400";
                              return (
                                <div key={idx} className={`${lineCl} whitespace-pre-wrap wrap-break-word`}>
                                  <span className="text-slate-500 select-none">[{ev.ts ? new Date(ev.ts).toLocaleTimeString() : ""}]</span>{" "}
                                  <span className="text-slate-400">[{ev.step_name || ev.stage || "info"}]</span>{" "}
                                  {ev.message}
                                </div>
                              );
                            }) : null}
                            {/* Fallback: WebSocket live logs */}
                            {liveLogs.length > 0 && !runState?.events?.length ? liveLogs.slice(-50).map((log: any, idx: number) => {
                              const isError = log.message?.includes("❌") || log.message?.includes("failed");
                              const isSuccess = log.message?.includes("✅") || log.message?.includes("succeeded");
                              const lineCl = isError ? "text-red-400" : isSuccess ? "text-emerald-300" : "text-green-400";
                              return (
                                <div key={idx} className={`${lineCl} whitespace-pre-wrap wrap-break-word`}>
                                  {log.stage ? `[${log.stage}] ` : ""}{log.message}
                                </div>
                              );
                            }) : null}
                            {/* Always show backend raw tail when training (real-time file read) */}
                            {training && backendLogTail.length > 0 && (
                              <>
                                <div className="text-slate-500 mt-2 pt-2 border-t border-slate-700"># backend.log (live)</div>
                                {backendLogTail.slice(-40).map((line: string, idx: number) => (
                                  <div key={`raw-${idx}`} className="text-slate-400 whitespace-pre-wrap wrap-break-word">
                                    {line}
                                  </div>
                                ))}
                              </>
                            )}
                            {training && (
                              <div className="text-amber-400 mt-1 flex items-center gap-1">
                                <span className="inline-block w-2 h-3 bg-amber-400 animate-pulse" />
                                {logStreamStatus === "streaming" ? "Live" : "Connecting…"}
                              </div>
                            )}
                            <div ref={logsEndRef} />
                          </div>
                        ) : (
                          <div className="text-slate-500 py-6 text-center">
                            {training ? (
                              <>
                                <div className="inline-block w-3 h-4 bg-green-500 animate-pulse mb-2" />
                                <p>{logStreamStatus === "connecting" ? "Connecting to run…" : "Waiting for stream…"} Logs will appear here in real time.</p>
                                {!backendOnline && (
                                  <p className="text-amber-400 mt-2 text-xs">Backend may be offline. Check that it is running on port 8000.</p>
                                )}
                              </>
                            ) : (
                              <p>Start training to see live logs.</p>
                            )}
                          </div>
                        )}
                      </div>

                      {/* What happened? — compact summary (same events, shown below terminal) */}
                      {(currentRunId && runState?.events?.length) ? (
                        <div className="space-y-1 mt-2">
                          <p className="text-xs text-muted-foreground">
                            Status: <strong>{runState.status}</strong> · Attempts: <strong>{runState.attempt_count}</strong> · Step: {runState.current_step || "—"}
                          </p>
                        </div>
                      ) : null}
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
                  {!backendOnline && (
                    <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-300">
                      Backend offline / restarting — UI will reconnect automatically.
                    </div>
                  )}
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

        {step === 4 && !sessionId && (
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

              {/* Notebook (live reasoning surface — attempt-based log) */}
              {(trainedModel?.run_id || currentRunId) && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Notebook (live reasoning surface)</CardTitle>
                    <CardDescription>Attempt-based log: Attempt 1 → Failure / Repair diff → Attempt 2 → Outcome. Append-only; no clean final report when refused.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <a
                      href={`${BACKEND_HTTP_BASE}/download/${trainedModel?.run_id || currentRunId}/notebook`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-primary hover:underline"
                    >
                      <FileJson className="w-4 h-4" />
                      Download notebook (.ipynb)
                    </a>
                  </CardContent>
                </Card>
              )}

              {/* What happened? — Execution timeline (visible on results step) */}
              {(step === 4 && (trainedModel?.run_id || currentRunId) && runState?.events?.length) ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">What happened?</CardTitle>
                    <CardDescription>Execution timeline: attempt_start, failure, repair_proposal (diff), retry.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-xs text-muted-foreground mb-2">
                      Status: {runState.status} · Attempts: {runState.attempt_count} · Step: {runState.current_step || "—"}
                    </div>
                    <div className="relative border-l-2 border-muted pl-3 space-y-1.5 max-h-[320px] overflow-y-auto">
                      {runState.events.slice(-100).map((ev: any, idx: number) => {
                        const step = (ev.step_name || ev.stage || "").toLowerCase();
                        const isFailed = ev.status === "failed" || (ev.message && (ev.message.includes("❌") || ev.message.includes("failed") || ev.message.includes("REFUSED")));
                        const isAttemptStart = step.includes("attempt_start");
                        const isAttemptFailure = step.includes("attempt_failure");
                        const isRepairProposal = step.includes("repair_proposal");
                        const isDiagnose = step.includes("diagnose");
                        const isRetry = step.includes("retry");
                        const isRepair = step.includes("repair");
                        return (
                          <div
                            key={idx}
                            className={`text-xs pl-2 py-1 rounded-r border-l-2 ${
                              isFailed || isAttemptFailure ? "border-red-500 bg-red-50 dark:bg-red-950/20" :
                              isAttemptStart ? "border-green-500 bg-green-50 dark:bg-green-950/20" :
                              isRepairProposal || isRepair ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20" :
                              isDiagnose ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20" :
                              isRetry ? "border-purple-500 bg-purple-50 dark:bg-purple-950/20" :
                              "border-muted bg-background"
                            }`}
                          >
                            <span className="font-medium text-muted-foreground">{ev.step_name || ev.stage || "info"}</span>
                            <span className="ml-1">{ev.message}</span>
                            {ev.payload && Object.keys(ev.payload).length > 0 && (
                              <details className="mt-1">
                                <summary className="cursor-pointer text-muted-foreground hover:underline">Payload</summary>
                                <pre className="mt-1 p-2 rounded bg-muted/50 text-[10px] overflow-x-auto whitespace-pre-wrap">
                                  {JSON.stringify(ev.payload, null, 1)}
                                </pre>
                              </details>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              ) : null}
              
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
                        {/* Primary metric (CV) - matches table so no discrepancy */}
                        {model.primary_metric != null && (
                          <div className="p-2 rounded bg-primary/10 border border-primary/20">
                            <p className="text-xs text-muted-foreground">Primary metric (CV mean)</p>
                            <p className="font-bold text-lg">{typeof model.primary_metric === 'number' ? model.primary_metric.toFixed(4) : model.primary_metric}</p>
                            <p className="text-xs text-muted-foreground">Same as table — cross-validation score</p>
                          </div>
                        )}
                        {/* Overfitting / performance note when CV << in-sample */}
                        {(() => {
                          const cvScore = model.primary_metric != null ? Number(model.primary_metric) : null;
                          const inSampleR2 = model.metrics?.r2 != null ? Number(model.metrics.r2) : null;
                          const inSampleAcc = model.metrics?.accuracy != null ? Number(model.metrics.accuracy) : null;
                          const gap = inSampleR2 != null && cvScore != null ? inSampleR2 - cvScore : (inSampleAcc != null && cvScore != null ? inSampleAcc - cvScore : null);
                          if (gap != null && gap > 0.15 && cvScore != null) {
                            return (
                              <div className="rounded-md border border-amber-500/50 bg-amber-50 dark:bg-amber-950/30 p-2 text-xs">
                                <p className="font-medium text-amber-800 dark:text-amber-200">⚠️ Why does performance look bad?</p>
                                <p className="text-amber-700 dark:text-amber-300 mt-0.5">
                                  CV score ({cvScore.toFixed(2)}) is much lower than in-sample ({inSampleR2 != null ? inSampleR2.toFixed(2) : inSampleAcc?.toFixed(2)}). 
                                  That usually means <strong>overfitting</strong> — the model fits the training data well but won’t generalize as well. 
                                  Real-world performance is better estimated by the <strong>CV (table)</strong> number.
                                </p>
                              </div>
                            );
                          }
                          return null;
                        })()}
                        <p className="text-xs font-medium text-muted-foreground">In-sample metrics</p>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(model.metrics || {})
                            .filter(([key]) => !String(key).startsWith("_"))
                            .slice(0, 4)
                            .map(([key, value]: [string, any]) => (
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
                        onClick={() => window.open(`${BACKEND_HTTP_BASE}/download/${trainedModel.run_id}/notebook`, '_blank')}
                      >
                        📓 Notebook
                      </Button>
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => window.open(`${BACKEND_HTTP_BASE}/download/${trainedModel.run_id}/model`, '_blank')}
                      >
                        💾 Model (.pkl)
                      </Button>
                      <Button 
                        variant="default" 
                        className="w-full bg-blue-600 hover:bg-blue-700"
                        onClick={() => window.open(`${BACKEND_HTTP_BASE}/download/${trainedModel.run_id}/report`, '_blank')}
                      >
                        📊 Full Report
                      </Button>
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => window.open(`${BACKEND_HTTP_BASE}/download/${trainedModel.run_id}/readme`, '_blank')}
                      >
                        📄 README
                      </Button>
                      <Button 
                        variant="secondary" 
                        className="w-full col-span-2 sm:col-span-1"
                        onClick={() => window.open(`${BACKEND_HTTP_BASE}/download/${trainedModel.run_id}/all`, '_blank')}
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
                            <p className="text-lg font-bold">
                              {predictionResult.prediction !== undefined && predictionResult.prediction !== null
                                ? String(predictionResult.prediction)
                                : "—"}
                            </p>
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
                    {selectedLlmProvider === "gemini_cli" 
                      ? "Configure LLM provider. CLI uses your Google account (no API key needed)."
                      : "Provide your Gemini API key to enable AI-powered features. The system will automatically switch models if rate limits are hit."}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {llmStatus && (
                    <div className="p-3 bg-muted rounded-md text-sm">
                      <div className="font-medium mb-1">Current Status:</div>
                      <div>
                        {selectedLlmProvider === "gemini_cli" ? (
                          <div>
                            {llmStatus.gemini_cli_available ? (
                              <>
                                ✅ <strong>CLI Active</strong> - Using: {llmStatus.gemini_cli_cmd || "gemini"} (local)
                                <div className="text-muted-foreground mt-1">
                                  Authenticated via Google account (OAuth)
                                </div>
                              </>
                            ) : (
                              <div>
                                ⚠️ <strong>CLI Not Found</strong>
                                <div className="text-muted-foreground mt-1">
                                  Gemini CLI not detected. Install it or switch to API mode.
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div>
                            {llmStatus.llm_available ? (
                              <>
                                ✅ <strong>API Active</strong> - Using: {llmStatus.current_model || "Default"}
                                {llmStatus.model_reason && (
                                  <div className="text-muted-foreground mt-1">
                                    {llmStatus.model_reason}
                                  </div>
                                )}
                              </>
                            ) : (
                              <div>⚠️ Using rule-based fallbacks</div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <Label htmlFor="llm-provider">LLM Provider</Label>
                    <select
                      id="llm-provider"
                      className="w-full h-10 rounded-md border bg-background px-3 text-sm"
                      value={selectedLlmProvider}
                      onChange={(e) => setSelectedLlmProvider(e.target.value)}
                    >
                      <option value="gemini">Gemini API (key-based)</option>
                      <option value="gemini_cli">Gemini CLI (local)</option>
                    </select>
                    {selectedLlmProvider === "gemini_cli" ? (
                      <p className="text-xs text-muted-foreground">
                        CLI uses Gemini auth via your Google account login (OAuth) by default.
                        {llmStatus?.gemini_cli_available ? (
                          <> CLI detected: <strong>yes</strong> {llmStatus?.gemini_cli_cmd ? `(cmd: ${llmStatus.gemini_cli_cmd})` : ""}</>
                        ) : (
                          <> CLI detected: <strong className="text-red-600">no</strong> - Install Gemini CLI or switch to API mode</>
                        )}
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        Uses your Gemini API key for authentication. The system will automatically switch models if rate limits are hit.
                      </p>
                    )}
                  </div>

                  {selectedLlmProvider === "gemini" && (
                    <div className="space-y-2">
                      <Label htmlFor="api-key">API Key (Gemini API)</Label>
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
                  )}

                  {selectedLlmProvider === "gemini_cli" && (
                    <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-md text-sm">
                      <div className="font-medium mb-1">ℹ️ CLI Mode</div>
                      <p className="text-xs text-muted-foreground">
                        No API key needed. CLI uses your Google account authentication.
                        {llmStatus?.gemini_cli_available ? (
                          <> CLI is ready to use.</>
                        ) : (
                          <> Install Gemini CLI or switch to API mode to use an API key.</>
                        )}
                      </p>
                    </div>
                  )}

                  {apiKeyStatus && selectedLlmProvider === "gemini" && (
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
                  {selectedLlmProvider === "gemini" && (
                    <Button
                      onClick={handleSetApiKey}
                      disabled={isSettingApiKey}
                    >
                      {isSettingApiKey ? "Setting..." : apiKey.trim() ? "Set Custom Key" : "Use Default Key"}
                    </Button>
                  )}
                </CardFooter>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
