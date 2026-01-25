'use client'

import { useState, useRef, useEffect } from 'react'
import { Upload, Send, Loader2 } from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [datasetId, setDatasetId] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
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
      
      addMessage('assistant', `hey, i analyzed your dataset. found ${data.profile.n_rows} rows and ${data.profile.n_cols} columns.`)
      
      if (data.profile.candidate_targets.length > 0) {
        addMessage('assistant', `looks like these might be what you want to predict: ${data.profile.candidate_targets.slice(0, 3).join(', ')}. which one?`)
      } else {
        addMessage('assistant', `what column do you want to predict?`)
      }
    } catch (error) {
      addMessage('assistant', 'oops, something went wrong uploading your file. try again?')
    } finally {
      setIsUploading(false)
    }
  }

  const addMessage = (role: 'user' | 'assistant' | 'system', content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      role,
      content,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
  }

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    addMessage('user', userMessage)
    setIsLoading(true)

    try {
      // Check if user is specifying target column
      if (!datasetId) {
        addMessage('assistant', 'upload a csv first, then we can chat about it')
        setIsLoading(false)
        return
      }

      // Simple intent detection
      const lowerInput = userMessage.toLowerCase()
      
      // If it looks like a target column name
      if (userMessage.length < 50 && !lowerInput.includes('?') && !lowerInput.includes('how') && !lowerInput.includes('what')) {
        // Try to train with this as target
        const response = await fetch('http://localhost:8000/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_id: datasetId,
            target: userMessage.trim(),
            task: 'classification', // Will be inferred
            metric: 'accuracy',
          }),
        })

        const data = await response.json()
        
        if (response.ok) {
          addMessage('assistant', `cool, trained a model. got ${(data.metrics.accuracy || data.metrics.r2 || 0).toFixed(3)} ${Object.keys(data.metrics)[0]}`)
          
          if (data.warnings && data.warnings.length > 0) {
            addMessage('assistant', `heads up: ${data.warnings[0]}`)
          }
          
          if (data.feature_importance) {
            const topFeatures = Object.entries(data.feature_importance)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .slice(0, 3)
              .map(([name]) => name)
            addMessage('assistant', `most important features: ${topFeatures.join(', ')}`)
          }
        } else {
          addMessage('assistant', `hmm, that didn't work. ${data.detail || 'try a different column?'}`)
        }
      } else {
        // Chat response - could integrate with explainer agent here
        addMessage('assistant', `yeah, i'm here to help with your ml pipeline. what do you want to know?`)
      }
    } catch (error) {
      addMessage('assistant', 'something broke. try again?')
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="min-h-screen bg-[#fafafa] flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-light text-gray-800">intent2model</h1>
          <p className="text-sm text-gray-500 mt-1">just upload and chat</p>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto max-w-3xl w-full mx-auto px-4 py-8">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-6">
            <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center">
              <Upload className="w-8 h-8 text-gray-400" />
            </div>
            <div>
              <h2 className="text-xl font-light text-gray-700 mb-2">drop your csv here</h2>
              <p className="text-sm text-gray-500">or click to browse</p>
            </div>
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                disabled={isUploading}
              />
              <div className="px-6 py-3 bg-gray-900 text-white rounded-full text-sm hover:bg-gray-800 transition-colors">
                {isUploading ? 'uploading...' : 'choose file'}
              </div>
            </label>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-gray-900 text-white'
                    : 'bg-white border border-gray-200 text-gray-800'
                }`}
              >
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="border-t border-gray-200 bg-white/80 backdrop-blur-sm sticky bottom-0">
        <div className="max-w-3xl mx-auto px-4 py-4">
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
                <Upload className="w-5 h-5 text-gray-400 hover:text-gray-600 transition-colors" />
              </label>
            )}
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={datasetId ? "ask me anything..." : "upload a csv first"}
              disabled={isLoading || !datasetId}
              className="flex-1 bg-gray-50 border border-gray-200 rounded-full px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading || !datasetId}
              className="w-10 h-10 rounded-full bg-gray-900 text-white flex items-center justify-center hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  )
}
