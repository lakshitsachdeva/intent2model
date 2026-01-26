'use client'

import { useState, useRef, useEffect } from 'react'
import { Upload, Send, Loader2, Sparkles, CheckCircle2, AlertCircle, BarChart3, TrendingUp, Zap, FileText, X, Brain, Database, Target, Activity } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  type?: 'info' | 'success' | 'error' | 'training' | 'chart' | 'prediction'
  chartData?: any
  modelData?: any
}

interface ModelResult {
  metrics: Record<string, number>
  feature_importance?: Record<string, number>
  warnings?: string[]
  run_id?: string
  cv_scores?: number[]
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [datasetId, setDatasetId] = useState<string | null>(null)
  const [availableColumns, setAvailableColumns] = useState<string[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [trainedModel, setTrainedModel] = useState<ModelResult | null>(null)
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [featureColumns, setFeatureColumns] = useState<string[]>([])
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictionFeatures, setPredictionFeatures] = useState<Record<string, string>>({})
  const [showPredictionModal, setShowPredictionModal] = useState(false)
  const [messageCounter, setMessageCounter] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      setDatasetId(data.dataset_id)
      setAvailableColumns(data.profile.numeric_cols.concat(data.profile.categorical_cols))
      
      addMessage('assistant', `✓ Analyzed your dataset`, 'info')
      addMessage('assistant', `${data.profile.n_rows} rows • ${data.profile.n_cols} columns`, 'info')
      
      if (data.profile.candidate_targets.length > 0) {
        addMessage('assistant', `Suggested targets: ${data.profile.candidate_targets.slice(0, 3).join(', ')}`, 'info')
        addMessage('assistant', `Which column should I predict?`, 'info')
      } else {
        addMessage('assistant', `What column do you want to predict?`, 'info')
      }
    } catch (error) {
      addMessage('assistant', 'Upload failed. Try again?', 'error')
    } finally {
      setIsUploading(false)
    }
  }

  const addMessage = (role: 'user' | 'assistant' | 'system', content: string, type: 'info' | 'success' | 'error' | 'training' | 'chart' | 'prediction' = 'info', chartData?: any, modelData?: any) => {
    const newMessage: Message = {
      id: `${Date.now()}-${messageCounter}-${Math.random().toString(36).substr(2, 9)}`,
      role,
      content,
      timestamp: new Date(),
      type,
      chartData,
      modelData,
    }
    setMessageCounter(prev => prev + 1)
    setMessages((prev) => [...prev, newMessage])
  }

  const isColumnName = (text: string): boolean => {
    const lowerText = text.toLowerCase().trim()
    return availableColumns.some(col => col.toLowerCase() === lowerText)
  }

  const isNaturalLanguageQuery = (text: string): boolean => {
    const lowerText = text.toLowerCase().trim()
    const queryKeywords = [
      'report', 'summary', 'explain', 'tell me', 'what', 'how', 'why', 'show me',
      'give me', 'make', 'create', 'build', 'train', 'all features', 'everything',
      'help', '?', 'describe', 'analyze', 'results', 'predict', 'prediction'
    ]
    return queryKeywords.some(keyword => lowerText.includes(keyword)) || 
           text.length > 30 ||
           text.split(' ').length > 5
  }

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    addMessage('user', userMessage)
    setIsLoading(true)

    try {
      if (!datasetId) {
        addMessage('assistant', 'Upload a CSV first', 'info')
        setIsLoading(false)
        return
      }

      const lowerInput = userMessage.toLowerCase().trim()

      if (lowerInput.includes('predict') || lowerInput.includes('what would') || lowerInput.includes('if i tell you') || lowerInput.includes('yes lets try') || lowerInput.includes('yes let')) {
        if (!trainedModel || !currentRunId) {
          addMessage('assistant', 'Train a model first, then I can make predictions', 'info')
          setIsLoading(false)
          return
        }
        setShowPredictionModal(true)
        setIsLoading(false)
        return
      }

      if (trainedModel && currentRunId && (userMessage.includes(':') || userMessage.match(/\d/))) {
        await handlePrediction(userMessage)
        setIsLoading(false)
        return
      }

      if (isNaturalLanguageQuery(userMessage)) {
        if (lowerInput.includes('report') || lowerInput.includes('summary') || lowerInput.includes('results')) {
          if (trainedModel) {
            showModelReport()
          } else {
            addMessage('assistant', 'Train a model first by telling me which column to predict', 'info')
          }
        } else if (lowerInput.includes('make') || lowerInput.includes('train') || lowerInput.includes('build')) {
          if (availableColumns.length > 0) {
            const target = availableColumns[0]
            await trainModel(target)
          } else {
            addMessage('assistant', 'Tell me which column to predict', 'info')
          }
        } else if (lowerInput.includes('all features') || lowerInput.includes('columns')) {
          addMessage('assistant', `Columns: ${availableColumns.join(', ')}`, 'info')
        } else {
          addMessage('assistant', 'Try: "train model" or tell me a column name', 'info')
        }
      } 
      else if (isColumnName(userMessage)) {
        await trainModel(userMessage.trim())
      }
      else {
        await trainModel(userMessage.trim())
      }
    } catch (error) {
      addMessage('assistant', 'Something went wrong. Try again?', 'error')
    } finally {
      setIsLoading(false)
    }
  }

  const handlePrediction = async (userInput?: string) => {
    setIsPredicting(true)
    try {
      let features: Record<string, any> = {}
      
      if (userInput) {
        try {
          const parseResponse = await fetch('http://localhost:8000/parse-prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_input: userInput,
              feature_columns: featureColumns,
              run_id: currentRunId
            }),
          })
          
          const parseData = await parseResponse.json()
          
          if (parseData.complete) {
            features = parseData.features
          } else {
            features = parseData.features
            const missing = parseData.missing || []
            if (missing.length > 0) {
              addMessage('assistant', `Still need: ${missing.join(', ')}`, 'info')
              setIsPredicting(false)
              return
            }
          }
        } catch (parseError) {
          addMessage('assistant', 'Could not parse your input. Please use the form or try again.', 'error')
          setShowPredictionModal(true)
          setIsPredicting(false)
          return
        }
      } else {
        features = {}
        featureColumns.forEach(col => {
          const value = predictionFeatures[col]?.trim()
          if (value) {
            const numValue = parseFloat(value)
            features[col] = isNaN(numValue) ? value : numValue
          }
        })
      }

      const missing = featureColumns.filter(col => !features[col] && features[col] !== 0)
      if (missing.length > 0) {
        addMessage('assistant', `Still need: ${missing.join(', ')}`, 'info')
        setIsPredicting(false)
        return
      }

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_id: currentRunId,
          features: features
        }),
      })

      const data = await response.json()
      
      if (response.ok) {
        setShowPredictionModal(false)
        addMessage('assistant', `Prediction: ${data.prediction}`, 'prediction', null, data)
        if (data.probabilities) {
          const probText = Object.entries(data.probabilities)
            .map(([k, v]) => `${k}: ${((v as number) * 100).toFixed(1)}%`)
            .join('\n')
          addMessage('assistant', `Probabilities:\n${probText}`, 'info')
        }
      } else {
        addMessage('assistant', data.detail || 'Prediction failed', 'error')
      }
    } catch (error) {
      addMessage('assistant', 'Prediction failed. Check your input?', 'error')
    } finally {
      setIsPredicting(false)
    }
  }

  const trainModel = async (target: string) => {
    addMessage('assistant', `Training model to predict "${target}"...`, 'training')
    
    try {
      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          target: target,
        }),
      })

      const data = await response.json()
      
      if (response.ok) {
        setTrainedModel(data)
        setCurrentRunId(data.run_id)
        const features = availableColumns.filter(col => col !== target)
        setFeatureColumns(features)
        
        addMessage('assistant', `✓ Model trained successfully!`, 'success')
        showCharts(data)
        
        if (data.model_comparison) {
          addMessage('assistant', `Tried ${data.model_comparison.tried_models.length} models, best: ${data.model_comparison.best_model}`, 'info')
        }
        
        if (data.pipeline_config) {
          const preproc = data.pipeline_config.preprocessing?.join(', ') || 'none'
          addMessage('assistant', `Pipeline: ${preproc} → ${data.pipeline_config.model}`, 'info')
        }
        
        if (data.warnings && data.warnings.length > 0) {
          addMessage('assistant', `Note: ${data.warnings[0]}`, 'info')
        }
        
        addMessage('assistant', `Want to make predictions? Just ask!`, 'info')
      } else {
        addMessage('assistant', data.detail || `Couldn't train with "${target}"`, 'error')
      }
    } catch (error) {
      addMessage('assistant', 'Training failed. Check your data?', 'error')
    }
  }

  const showCharts = (data: ModelResult) => {
    if (data.metrics) {
      const metricsData = Object.entries(data.metrics)
        .filter(([_, v]) => v !== null && v !== undefined)
        .map(([name, value]) => ({
          name: name.toUpperCase(),
          value: Number(value)
        }))
      
      addMessage('assistant', 'Performance Metrics', 'chart', { type: 'metrics', data: metricsData }, data)
    }

    if (data.feature_importance) {
      const featureData = Object.entries(data.feature_importance)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 10)
        .map(([name, importance]) => ({
          name: name.length > 20 ? name.substring(0, 20) + '...' : name,
          importance: Number(importance)
        }))
      
      addMessage('assistant', 'Feature Importance', 'chart', { type: 'features', data: featureData }, data)
    }

    if (data.cv_scores && data.cv_scores.length > 0) {
      const cvData = data.cv_scores.map((score, i) => ({
        fold: `Fold ${i + 1}`,
        score: Number(score)
      }))
      addMessage('assistant', 'Cross-Validation Scores', 'chart', { type: 'cv', data: cvData }, data)
    }
  }

  const showModelReport = () => {
    if (!trainedModel) return
    addMessage('assistant', 'Model Report:', 'success')
    showCharts(trainedModel)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const renderChart = (chartData: any) => {
    if (!chartData || !chartData.data) return null

    if (chartData.type === 'metrics') {
      return (
        <div className="mt-6 w-full bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-100">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            Performance Metrics
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#fff', 
                  border: '2px solid #e2e8f0',
                  borderRadius: '12px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }} 
              />
              <Bar dataKey="value" fill="#6366f1" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )
    }

    if (chartData.type === 'features') {
      return (
        <div className="mt-6 w-full bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-xl border border-purple-100">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-purple-600" />
            Feature Importance
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData.data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis type="number" stroke="#64748b" fontSize={12} />
              <YAxis dataKey="name" type="category" width={140} stroke="#64748b" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#fff', 
                  border: '2px solid #e2e8f0',
                  borderRadius: '12px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }} 
              />
              <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )
    }

    if (chartData.type === 'cv') {
      return (
        <div className="mt-6 w-full bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-xl border border-green-100">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-600" />
            Cross-Validation Scores
          </h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis dataKey="fold" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#fff', 
                  border: '2px solid #e2e8f0',
                  borderRadius: '12px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }} 
              />
              <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={3} dot={{ fill: '#10b981', r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )
    }

    return null
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex flex-col">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 sticky top-0 z-20 shadow-lg">
        <div className="max-w-7xl mx-auto px-8 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 flex items-center justify-center shadow-lg">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Intent2Model</h1>
                <p className="text-sm text-gray-600 font-medium">LLM-Guided AutoML Platform</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {datasetId && (
                <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
                  <Database className="w-4 h-4 text-blue-600" />
                  <span className="text-sm font-semibold text-blue-700">Dataset Loaded</span>
                </div>
              )}
              {trainedModel && (
                <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 rounded-lg shadow-sm">
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                  <span className="text-sm font-bold text-green-700">Model Ready</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex max-w-7xl mx-auto w-full gap-6 p-6">
        {/* Sidebar */}
        <aside className="w-80 flex-shrink-0 space-y-4">
          <div className="bg-white rounded-2xl shadow-xl border border-gray-200 p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-indigo-600" />
              Quick Actions
            </h2>
            <div className="space-y-3">
              {!datasetId && (
                <label className="block">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={isUploading}
                  />
                  <div className="w-full px-4 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl text-sm font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl cursor-pointer text-center">
                    {isUploading ? 'Uploading...' : '📁 Upload Dataset'}
                  </div>
                </label>
              )}
              {trainedModel && (
                <button
                  onClick={() => setShowPredictionModal(true)}
                  className="w-full px-4 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl text-sm font-semibold hover:from-green-700 hover:to-emerald-700 transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                >
                  <Zap className="w-4 h-4" />
                  Make Prediction
                </button>
              )}
              {trainedModel && (
                <button
                  onClick={showModelReport}
                  className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl text-sm font-semibold hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                >
                  <BarChart3 className="w-4 h-4" />
                  View Report
                </button>
              )}
            </div>
          </div>

          {trainedModel && (
            <div className="bg-white rounded-2xl shadow-xl border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Model Metrics</h3>
              <div className="space-y-3">
                {Object.entries(trainedModel.metrics || {}).slice(0, 4).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm font-medium text-gray-700 capitalize">{key}</span>
                    <span className="text-sm font-bold text-indigo-600">{typeof value === 'number' ? value.toFixed(3) : value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* Chat Area */}
        <main className="flex-1 flex flex-col bg-white rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-gray-50 to-white">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center space-y-8 py-20">
                <div className="w-32 h-32 rounded-3xl bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 flex items-center justify-center shadow-2xl">
                  <FileText className="w-16 h-16 text-white" />
                </div>
                <div className="space-y-3">
                  <h2 className="text-3xl font-bold text-gray-900">Get Started with AutoML</h2>
                  <p className="text-gray-600 max-w-md">Upload your dataset and let AI build the perfect machine learning model for you</p>
                </div>
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={isUploading}
                  />
                  <div className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl text-base font-bold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-xl hover:shadow-2xl hover:scale-105">
                    {isUploading ? 'Uploading...' : '📁 Upload CSV File'}
                  </div>
                </label>
              </div>
            )}

            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-in`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-6 py-4 shadow-lg ${
                      message.role === 'user'
                        ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                        : message.type === 'error'
                        ? 'bg-red-50 border-2 border-red-300 text-red-900'
                        : message.type === 'success'
                        ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 text-green-900'
                        : message.type === 'training'
                        ? 'bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-300 text-blue-900'
                        : message.type === 'chart'
                        ? 'bg-white border-2 border-gray-200 shadow-xl'
                        : message.type === 'prediction'
                        ? 'bg-gradient-to-r from-yellow-50 to-amber-50 border-2 border-yellow-300 text-amber-900'
                        : 'bg-white border-2 border-gray-200 text-gray-900'
                    }`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap font-medium">{message.content}</p>
                    {message.type === 'chart' && message.chartData && renderChart(message.chartData)}
                    {message.type === 'prediction' && message.modelData && (
                      <div className="mt-4 p-4 bg-gradient-to-r from-yellow-100 to-amber-100 rounded-xl border-2 border-yellow-400 shadow-md">
                        <div className="text-2xl font-bold text-yellow-900 mb-2">{message.modelData.prediction}</div>
                        {message.modelData.probabilities && (
                          <div className="mt-3 space-y-2">
                            {Object.entries(message.modelData.probabilities).map(([label, prob]) => (
                              <div key={label} className="flex justify-between items-center p-2 bg-white/60 rounded-lg">
                                <span className="text-sm font-medium text-gray-800">{label}</span>
                                <span className="text-sm font-bold text-indigo-600">{((prob as number) * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border-2 border-gray-200 rounded-2xl px-6 py-4 shadow-lg">
                    <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 bg-white p-6">
            <div className="flex items-center gap-4">
              {!datasetId && (
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={isUploading}
                  />
                  <div className="p-3 rounded-xl hover:bg-indigo-50 transition-colors border-2 border-indigo-200">
                    <Upload className="w-6 h-6 text-indigo-600" />
                  </div>
                </label>
              )}
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={datasetId ? (trainedModel ? "Ask me to predict something..." : "Tell me which column to predict...") : "Upload a CSV first"}
                disabled={isLoading || !datasetId}
                className="flex-1 bg-gray-50 border-2 border-gray-300 rounded-xl px-6 py-4 text-base focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:opacity-50 font-medium"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading || !datasetId}
                className="p-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
              >
                <Send className="w-6 h-6" />
              </button>
            </div>
          </div>
        </main>
        </div>

      {/* Prediction Modal */}
      {showPredictionModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col border-2 border-gray-200">
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6 flex items-center justify-between">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Zap className="w-6 h-6" />
                Enter Feature Values
              </h2>
              <button
                onClick={() => setShowPredictionModal(false)}
                className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              >
                <X className="w-6 h-6 text-white" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-8 space-y-5 bg-gray-50">
              {featureColumns.map((col) => (
                <div key={col} className="bg-white rounded-xl p-5 border-2 border-gray-200 shadow-sm">
                  <label className="block text-base font-bold text-gray-900 mb-3">
                    {col}
                  </label>
                  <input
                    type="text"
                    value={predictionFeatures[col] || ''}
                    onChange={(e) => setPredictionFeatures({ ...predictionFeatures, [col]: e.target.value })}
                    placeholder="Enter value..."
                    className="w-full px-5 py-3 border-2 border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-base font-medium"
                  />
                </div>
              ))}
            </div>
            <div className="bg-gray-50 border-t-2 border-gray-200 px-8 py-6 flex items-center justify-end gap-4">
              <button
                onClick={() => setShowPredictionModal(false)}
                className="px-6 py-3 text-base font-semibold text-gray-700 hover:bg-gray-200 rounded-xl transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handlePrediction()}
                disabled={isPredicting || featureColumns.some(col => !predictionFeatures[col]?.trim())}
                className="px-6 py-3 text-base font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg"
              >
                {isPredicting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Predict
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
