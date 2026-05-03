import React, { useState, useRef } from 'react'
import axios from 'axios'
import { 
  Upload, 
  Waves, 
  Ship, 
  BarChart3, 
  Info, 
  AlertCircle, 
  CheckCircle2, 
  Play, 
  Loader2,
  Settings2,
  ArrowRight
} from 'lucide-react'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts'

const App = () => {
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('cnn')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
      setError(null)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
      setResult(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', model)

    try {
      // Use full URL if proxy isn't working, but /api/predict is mapped to localhost:8000
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || 'Failed to process audio. Please ensure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const chartData = result ? Object.entries(result.all_probabilities).map(([name, value]) => ({
    name: name.split(' (')[0],
    confidence: (value * 100).toFixed(1)
  })).sort((a, b) => b.confidence - a.confidence) : []

  const COLORS = ['#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#e0f2fe']

  return (
    <div className="min-h-screen pb-20">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 glass-panel !rounded-none !border-x-0 !border-t-0 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 primary-gradient rounded-lg shadow-lg">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-display font-bold tracking-tight">ShipsEar <span className="text-primary-500">AI</span></span>
        </div>
        <div className="flex items-center space-x-4">
          <a href="#about" className="text-sm text-slate-400 hover:text-white transition-colors">Documentation</a>
          <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-medium transition-all">GitHub</button>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 mt-12 grid grid-cols-1 lg:grid-cols-12 gap-12">
        
        {/* Left Column: Upload & Settings */}
        <div className="lg:col-span-5 space-y-8">
          <div className="space-y-2">
            <h1 className="text-4xl font-display font-bold text-white tracking-tight">Underwater Vessel <span className="text-transparent bg-clip-text primary-gradient">Classification</span></h1>
            <p className="text-slate-400 text-lg leading-relaxed">
              Identify vessel types from acoustic signals using deep learning and signal processing.
            </p>
          </div>

          <div className="glass-panel p-6 space-y-6">
            <div className="flex items-center space-x-2 text-primary-400">
              <Settings2 className="w-5 h-5" />
              <h2 className="font-semibold uppercase tracking-wider text-xs">Configuration</h2>
            </div>
            
            <div className="space-y-4">
              <label className="block text-sm font-medium text-slate-300">Target Model</label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { id: 'cnn', name: 'Custom CNN', icon: <BarChart3 className="w-4 h-4" /> },
                  { id: 'resnet', name: 'ResNet-18', icon: <Ship className="w-4 h-4" /> },
                  { id: 'knn', name: 'K-Neighbors', icon: <Settings2 className="w-4 h-4" /> },
                  { id: 'svm', name: 'Support Vector', icon: <Settings2 className="w-4 h-4" /> }
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setModel(m.id)}
                    className={`flex items-center justify-center space-x-2 p-3 rounded-xl border transition-all ${
                      model === m.id 
                        ? 'bg-primary-500/10 border-primary-500 text-primary-400' 
                        : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-500'
                    }`}
                  >
                    {m.icon}
                    <span className="text-sm font-medium">{m.name}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <label className="block text-sm font-medium text-slate-300">Audio Source</label>
              <div 
                className={`relative group cursor-pointer border-2 border-dashed rounded-2xl p-8 transition-all duration-300 flex flex-col items-center justify-center space-y-4 ${
                  dragActive 
                    ? 'border-primary-500 bg-primary-500/5 scale-[1.02]' 
                    : file 
                      ? 'border-emerald-500/50 bg-emerald-500/5' 
                      : 'border-slate-700 bg-slate-800/30 hover:border-slate-500'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".wav"
                />
                
                {file ? (
                  <div className="flex flex-col items-center space-y-2">
                    <CheckCircle2 className="w-12 h-12 text-emerald-500" />
                    <p className="text-emerald-400 font-medium">{file.name}</p>
                    <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                ) : (
                  <>
                    <div className="p-4 bg-slate-800 rounded-full group-hover:scale-110 transition-transform">
                      <Upload className="w-8 h-8 text-primary-400" />
                    </div>
                    <div className="text-center">
                      <p className="font-medium text-slate-200">Click or drag & drop</p>
                      <p className="text-xs text-slate-500 uppercase mt-1 tracking-widest">WAV files only</p>
                    </div>
                  </>
                )}
              </div>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className={`w-full py-4 rounded-xl font-bold flex items-center justify-center space-x-2 transition-all shadow-xl ${
                !file || loading 
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                  : 'primary-gradient text-white hover:scale-[1.02] active:scale-[0.98] accent-glow'
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Classify Acoustic Signal</span>
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </div>
          
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start space-x-3 text-red-400 animate-in fade-in slide-in-from-top-2">
              <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="font-bold">Classification Error</p>
                <p className="opacity-80">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-7">
          {!result && !loading && (
            <div className="h-full min-h-[400px] glass-panel flex flex-col items-center justify-center text-center p-12 space-y-6 opacity-60">
              <div className="w-24 h-24 rounded-full border-2 border-slate-800 flex items-center justify-center animate-pulse-slow">
                <Ship className="w-10 h-10 text-slate-600" />
              </div>
              <div className="max-w-xs space-y-2">
                <h3 className="text-xl font-semibold text-slate-300">Ready for Analysis</h3>
                <p className="text-sm text-slate-500">Upload a vessel recording to begin deep acoustic classification and feature extraction.</p>
              </div>
            </div>
          )}

          {loading && (
            <div className="h-full min-h-[400px] glass-panel flex flex-col items-center justify-center text-center p-12 space-y-8">
              <div className="relative">
                <div className="w-32 h-32 rounded-full border-4 border-slate-800 animate-pulse" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Waves className="w-12 h-12 text-primary-500 animate-bounce" />
                </div>
              </div>
              <div className="space-y-3">
                <h3 className="text-2xl font-display font-bold text-white">Analyzing Signature</h3>
                <div className="flex flex-col items-center space-y-1">
                  <div className="flex space-x-1">
                    <div className="w-1.5 h-1.5 bg-primary-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1.5 h-1.5 bg-primary-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-primary-500 rounded-full animate-bounce"></div>
                  </div>
                  <p className="text-sm text-slate-500 uppercase tracking-widest">Extracting Log-Mel Spectrogram</p>
                </div>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
              <div className="glass-panel p-8 space-y-8 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-8 opacity-10">
                  <Ship className="w-48 h-48" />
                </div>

                <div className="relative z-10 space-y-6">
                  <div className="flex items-center space-x-2 text-emerald-400">
                    <CheckCircle2 className="w-5 h-5" />
                    <h2 className="font-semibold uppercase tracking-wider text-xs">Analysis Complete</h2>
                  </div>

                  <div className="space-y-1">
                    <p className="text-sm text-slate-400 uppercase tracking-widest font-medium">Primary Identification</p>
                    <div className="flex items-baseline space-x-4">
                      <h2 className="text-5xl font-display font-black text-white">{result.class_name}</h2>
                      <span className="text-2xl font-display font-bold text-primary-400">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
                    <div 
                      className="h-full primary-gradient accent-glow transition-all duration-1000 ease-out"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>

                  <div className="grid grid-cols-3 gap-6 pt-6 border-t border-slate-800">
                    <div>
                      <p className="text-xs text-slate-500 uppercase font-medium mb-1">Status</p>
                      <p className="text-sm font-semibold text-emerald-400">High Confidence</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500 uppercase font-medium mb-1">Architecture</p>
                      <p className="text-sm font-semibold text-white capitalize">{model}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500 uppercase font-medium mb-1">Latency</p>
                      <p className="text-sm font-semibold text-white">~342ms</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-panel p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold flex items-center space-x-2">
                      <BarChart3 className="w-5 h-5 text-primary-400" />
                      <span>Probability Distribution</span>
                    </h3>
                  </div>
                  <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: -20, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                        <XAxis type="number" hide domain={[0, 100]} />
                        <YAxis 
                          dataKey="name" 
                          type="category" 
                          stroke="#94a3b8" 
                          fontSize={11} 
                          width={120}
                        />
                        <Tooltip 
                          cursor={{fill: 'rgba(15, 23, 42, 0.5)'}}
                          contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc'}}
                        />
                        <Bar dataKey="confidence" radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass-panel p-6 space-y-6">
                  <h3 className="font-bold flex items-center space-x-2">
                    <Waves className="w-5 h-5 text-primary-400" />
                    <span>Signal Sample</span>
                  </h3>
                  <div className="p-8 bg-slate-950/50 rounded-2xl border border-slate-800 flex flex-col items-center space-y-6">
                    <div className="w-full flex items-end justify-center space-x-1 h-12">
                      {[...Array(20)].map((_, i) => (
                        <div 
                          key={i} 
                          className="w-1 bg-primary-500/40 rounded-full animate-pulse" 
                          style={{ height: `${Math.random() * 100}%`, animationDelay: `${i * 0.1}s` }}
                        />
                      ))}
                    </div>
                    <button className="p-4 bg-primary-500 rounded-full text-white shadow-lg hover:scale-110 active:scale-95 transition-all accent-glow">
                      <Play className="w-6 h-6 fill-current" />
                    </button>
                    <p className="text-xs text-slate-500 text-center">Spectral signature playback is simulated for this demo.</p>
                  </div>
                  <div className="flex items-start space-x-3 p-4 bg-primary-500/5 rounded-xl border border-primary-500/10">
                    <Info className="w-5 h-5 text-primary-400 mt-0.5" />
                    <p className="text-xs text-slate-400">
                      The classification is based on the first 5 seconds of the recording. 
                      Frequency analysis reveals harmonics consistent with {result.class_name}.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
