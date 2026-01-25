'use client'

import { useState, useRef, useEffect } from 'react'
import { Upload, Send, Loader2, Sparkles, CheckCircle2, AlertCircle, BarChart3, TrendingUp, Zap } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'

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
      
      addMessage('assistant', `analyzed your dataset`, 'info')
      addMessage('assistant', `${data.profile.n_rows} rows • ${data.profile.n_cols} columns`, 'info')
      
      if (data.profile.candidate_targets.length > 0) {
        addMessage('assistant', `suggested targets: ${data.profile.candidate_targets.slice(0, 3).join(', ')}`, 'info')
        addMessage('assistant', `which column should i predict?`, 'info')
      } else {
        addMessage('assistant', `what column do you want to predict?`, 'info')
      }
    } catch (error) {
      addMessage('assistant', 'upload failed. try again?', 'error')
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
        addMessage('assistant', 'upload a csv first', 'info')
        setIsLoading(false)
        return
      }

      const lowerInput = userMessage.toLowerCase().trim()

      // Check for prediction requests
      if (lowerInput.includes('predict') || lowerInput.includes('what would') || lowerInput.includes('if i tell you')) {
        if (!trainedModel || !currentRunId) {
          addMessage('assistant', 'train a model first, then i can make predictions', 'info')
          setIsLoading(false)
          return
        }
        // Start prediction flow
        addMessage('assistant', `sure! i need these features: ${featureColumns.join(', ')}. tell me the values one by one, or all at once like "sepal.length: 5.1, sepal.width: 3.5"`, 'info')
        setIsLoading(false)
        return
      }

      // Check if user is providing feature values for prediction
      if (trainedModel && currentRunId && (userMessage.includes(':') || userMessage.match(/\d/))) {
        await handlePrediction(userMessage)
        setIsLoading(false)
        return
      }

      // Check if it's a natural language query
      if (isNaturalLanguageQuery(userMessage)) {
        if (lowerInput.includes('report') || lowerInput.includes('summary') || lowerInput.includes('results')) {
          if (trainedModel) {
            showModelReport()
          } else {
            addMessage('assistant', 'train a model first by telling me which column to predict', 'info')
          }
        } else if (lowerInput.includes('make') || lowerInput.includes('train') || lowerInput.includes('build')) {
          if (availableColumns.length > 0) {
            const target = availableColumns[0]
            await trainModel(target)
          } else {
            addMessage('assistant', 'tell me which column to predict', 'info')
          }
        } else if (lowerInput.includes('all features') || lowerInput.includes('columns')) {
          addMessage('assistant', `columns: ${availableColumns.join(', ')}`, 'info')
        } else {
          addMessage('assistant', 'try: "train model" or tell me a column name', 'info')
        }
      } 
      // Check if it's a column name
      else if (isColumnName(userMessage)) {
        await trainModel(userMessage.trim())
      }
      // Try as column name
      else {
        await trainModel(userMessage.trim())
      }
    } catch (error) {
      addMessage('assistant', 'something went wrong. try again?', 'error')
    } finally {
      setIsLoading(false)
    }
  }

  const handlePrediction = async (userInput: string) => {
    setIsPredicting(true)
    try {
      // Parse feature values from input
      const features: Record<string, any> = {}
      
      // Try to parse "key: value" format
      const pairs = userInput.split(',').map(s => s.trim())
      for (const pair of pairs) {
        const [key, value] = pair.split(':').map(s => s.trim())
        if (key && value) {
          const numValue = parseFloat(value)
          features[key] = isNaN(numValue) ? value : numValue
        }
      }

      // Check if we have all required features
      const missing = featureColumns.filter(col => !features[col])
      if (missing.length > 0) {
        addMessage('assistant', `still need: ${missing.join(', ')}`, 'info')
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
        addMessage('assistant', `prediction: ${data.prediction}`, 'prediction', null, data)
        if (data.probabilities) {
          const probText = Object.entries(data.probabilities)
            .map(([k, v]) => `${k}: ${((v as number) * 100).toFixed(1)}%`)
            .join('\n')
          addMessage('assistant', `probabilities:\n${probText}`, 'info')
        }
      } else {
        addMessage('assistant', data.detail || 'prediction failed', 'error')
      }
    } catch (error) {
      addMessage('assistant', 'prediction failed. check your input?', 'error')
    } finally {
      setIsPredicting(false)
    }
  }

  const trainModel = async (target: string) => {
    addMessage('assistant', `training model to predict "${target}"...`, 'training')
    
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
        // Get feature columns (all columns except target)
        const features = availableColumns.filter(col => col !== target)
        setFeatureColumns(features)
        
        addMessage('assistant', `model trained successfully`, 'success')
        
        // Show charts
        showCharts(data)
        
        // Show model comparison if multiple models were tried
        if (data.model_comparison) {
          addMessage('assistant', `tried ${data.model_comparison.tried_models.length} models, best: ${data.model_comparison.best_model}`, 'info')
        }
        
        if (data.pipeline_config) {
          const preproc = data.pipeline_config.preprocessing?.join(', ') || 'none'
          addMessage('assistant', `pipeline: ${preproc} → ${data.pipeline_config.model}`, 'info')
        }
        
        if (data.warnings && data.warnings.length > 0) {
          addMessage('assistant', `note: ${data.warnings[0]}`, 'info')
        }
        
        addMessage('assistant', `want to make predictions? just ask!`, 'info')
      } else {
        addMessage('assistant', data.detail || `couldn't train with "${target}"`, 'error')
      }
    } catch (error) {
      addMessage('assistant', 'training failed. check your data?', 'error')
    }
  }

  const showCharts = (data: ModelResult) => {
    // Metrics bar chart
    if (data.metrics) {
      const metricsData = Object.entries(data.metrics)
        .filter(([_, v]) => v !== null && v !== undefined)
        .map(([name, value]) => ({
          name: name.toUpperCase(),
          value: Number(value)
        }))
      
      addMessage('assistant', 'performance metrics', 'chart', { type: 'metrics', data: metricsData }, data)
    }

    // Feature importance chart
    if (data.feature_importance) {
      const featureData = Object.entries(data.feature_importance)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 10)
        .map(([name, importance]) => ({
          name: name.length > 20 ? name.substring(0, 20) + '...' : name,
          importance: Number(importance)
        }))
      
      addMessage('assistant', 'feature importance', 'chart', { type: 'features', data: featureData }, data)
    }

    // CV scores line chart
    if (data.cv_scores && data.cv_scores.length > 0) {
      const cvData = data.cv_scores.map((score, i) => ({
        fold: `Fold ${i + 1}`,
        score: Number(score)
      }))
      addMessage('assistant', 'cross-validation scores', 'chart', { type: 'cv', data: cvData }, data)
    }
  }

  const showModelReport = () => {
    if (!trainedModel) return
    
    addMessage('assistant', 'model report:', 'success')
    showCharts(trainedModel)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'training':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
      case 'chart':
        return <BarChart3 className="w-4 h-4 text-purple-500" />
      case 'prediction':
        return <Zap className="w-4 h-4 text-yellow-500" />
      case 'info':
        return <BarChart3 className="w-4 h-4 text-gray-400" />
      default:
        return null
    }
  }

  const renderChart = (chartData: any) => {
    if (!chartData || !chartData.data) return null

    const COLORS = ['#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444', '#6366f1', '#14b8a6']

    if (chartData.type === 'metrics') {
      return (
        <div className="mt-4 w-full">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )
    }

    if (chartData.type === 'features') {
      return (
        <div className="mt-4 w-full">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData.data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={120} />
              <Tooltip />
              <Bar dataKey="importance" fill="#ec4899" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )
    }

    if (chartData.type === 'cv') {
      return (
        <div className="mt-4 w-full">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fold" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={3} dot={{ fill: '#10b981', r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )
    }

    return null
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 flex flex-col">
      {/* Header */}
      <header className="border-b border-purple-200/50 bg-white/90 backdrop-blur-xl sticky top-0 z-10 w-full shadow-lg">
        <div className="max-w-5xl mx-auto px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center shadow-xl">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">intent2model</h1>
              <p className="text-xs text-gray-600 mt-0.5">just upload and chat</p>
            </div>
            {trainedModel && (
              <div className="ml-auto flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 rounded-xl shadow-md">
                <CheckCircle2 className="w-5 h-5 text-green-600" />
                <span className="text-sm font-semibold text-green-700">model ready</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto max-w-5xl w-full mx-auto px-6 py-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-8 pt-20">
            <div className="w-32 h-32 rounded-3xl bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center shadow-2xl animate-pulse">
              <Upload className="w-16 h-16 text-white" />
            </div>
            <div className="space-y-3">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">drop your csv here</h2>
              <p className="text-sm text-gray-600">or click to browse</p>
            </div>
            <label className="cursor-pointer group">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                disabled={isUploading}
              />
              <div className="px-10 py-5 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-2xl text-sm font-bold hover:from-purple-700 hover:to-pink-700 transition-all shadow-xl hover:shadow-2xl group-hover:scale-110">
                {isUploading ? 'uploading...' : 'choose file'}
              </div>
            </label>
          </div>
        )}

        <div className="space-y-4 py-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in`}
            >
              {message.role === 'assistant' && message.type && (
                <div className="mr-3 mt-1">
                  {getMessageIcon(message.type)}
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-3xl px-6 py-4 shadow-lg ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-gray-900 to-gray-800 text-white'
                    : message.type === 'error'
                    ? 'bg-red-50 border-2 border-red-300 text-red-900'
                    : message.type === 'success'
                    ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 text-green-900'
                    : message.type === 'training'
                    ? 'bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-300 text-blue-900'
                    : message.type === 'chart'
                    ? 'bg-white border-2 border-purple-200 shadow-xl'
                    : message.type === 'prediction'
                    ? 'bg-gradient-to-r from-yellow-50 to-amber-50 border-2 border-yellow-300 text-amber-900'
                    : 'bg-white border-2 border-gray-200 text-gray-800'
                }`}
              >
                <p className="text-sm leading-relaxed whitespace-pre-wrap font-medium">{message.content}</p>
                {message.type === 'chart' && message.chartData && renderChart(message.chartData)}
                {message.type === 'prediction' && message.modelData && (
                  <div className="mt-3 p-3 bg-yellow-100 rounded-xl border border-yellow-300">
                    <div className="text-2xl font-bold text-yellow-900">{message.modelData.prediction}</div>
                    {message.modelData.probabilities && (
                      <div className="mt-2 space-y-1">
                        {Object.entries(message.modelData.probabilities).map(([label, prob]) => (
                          <div key={label} className="flex justify-between text-xs">
                            <span>{label}:</span>
                            <span className="font-bold">{((prob as number) * 100).toFixed(1)}%</span>
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
              <div className="mr-3 mt-1">
                <Loader2 className="w-4 h-4 animate-spin text-purple-500" />
              </div>
              <div className="bg-white border-2 border-purple-200 rounded-3xl px-6 py-4 shadow-lg">
                <Loader2 className="w-4 h-4 animate-spin text-purple-500" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="border-t border-purple-200/50 bg-white/90 backdrop-blur-xl sticky bottom-0 w-full shadow-2xl">
        <div className="max-w-5xl mx-auto px-6 py-5">
          <div className="flex items-center gap-3">
            {!datasetId && (
              <label className="cursor-pointer">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={isUploading}
                />
                <div className="p-3 rounded-xl hover:bg-purple-100 transition-colors">
                  <Upload className="w-5 h-5 text-purple-500" />
                </div>
              </label>
            )}
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={datasetId ? (trainedModel ? "ask me to predict something..." : "tell me which column to predict...") : "upload a csv first"}
              disabled={isLoading || !datasetId}
              className="flex-1 bg-white border-2 border-purple-200 rounded-2xl px-6 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 disabled:opacity-50 shadow-md"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading || !datasetId}
              className="p-4 rounded-2xl bg-gradient-to-r from-purple-600 to-pink-600 text-white flex items-center justify-center hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-xl hover:shadow-2xl disabled:hover:shadow-xl"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  )
}
