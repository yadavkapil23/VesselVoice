import React, { useRef, useState } from 'react'
import axios from 'axios'
import {
  AlertCircle,
  ArrowRight,
  AudioWaveform,
  BarChart3,
  CheckCircle2,
  FileAudio2,
  Info,
  Loader2,
  Radar,
  Settings2,
  Ship,
  Sparkles,
  Upload,
  Waves,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const modelOptions = [
  {
    id: 'cnn',
    name: 'Custom CNN',
    description: 'Fast deep pattern extraction for spectrogram-based predictions.',
    icon: Radar,
  },
  {
    id: 'resnet',
    name: 'ResNet-18',
    description: 'Transfer-learned visual encoder tuned for acoustic signatures.',
    icon: Ship,
  },
  {
    id: 'knn',
    name: 'K-Neighbors',
    description: 'Distance-based classification for handcrafted feature spaces.',
    icon: BarChart3,
  },
  {
    id: 'svm',
    name: 'Support Vector',
    description: 'Margin-optimized classification for structured signal features.',
    icon: Settings2,
  },
]

const waveformBars = [34, 52, 46, 68, 40, 74, 58, 84, 50, 64, 38, 72, 54, 88, 61, 76, 44, 66, 48, 70]

const pipelineSteps = [
  'Upload a `.wav` segment from the ShipsEar workflow.',
  'Choose the model you want to evaluate.',
  'Run inference and compare confidence across vessel classes.',
]

const insightCards = [
  { label: 'Input Type', value: 'Hydrophone WAV', icon: FileAudio2 },
  { label: 'Signal Window', value: 'First 5 Seconds', icon: AudioWaveform },
  { label: 'Output Mode', value: 'Multi-Class Ranking', icon: Sparkles },
]

const formatModelName = (value) => {
  const match = modelOptions.find((option) => option.id === value)
  return match ? match.name : value
}

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
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
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
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || 'Failed to process audio. Please ensure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const chartData = result
    ? Object.entries(result.all_probabilities)
        .map(([name, value]) => ({
          name: name.split(' (')[0],
          confidence: Number((value * 100).toFixed(1)),
        }))
        .sort((a, b) => b.confidence - a.confidence)
    : []

  const topConfidence = result ? Math.round(result.confidence * 100) : 0
  const resultTone =
    topConfidence >= 80 ? 'Strong match' : topConfidence >= 60 ? 'Moderate match' : 'Low-certainty match'
  const resultSummary = result
    ? `The selected ${formatModelName(model)} model ranked ${result.class_name} highest among the available vessel classes.`
    : null

  const COLORS = ['#17c3b2', '#2ec4b6', '#7adfcb', '#b7efe6', '#e3fbf7']

  return (
    <div className="min-h-screen text-white">
      <div className="page-shell">
        <nav className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-6 lg:px-10">
          <div className="flex items-center gap-3">
            <div className="brand-badge">
              <Waves className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.35em] text-cyan-200/70">ShipsEar Lab</p>
              <h1 className="text-lg font-semibold tracking-tight text-white">VesselVoice Console</h1>
            </div>
          </div>

          <div className="hidden items-center gap-3 md:flex">
            <a href="#workflow" className="nav-link">
              Workflow
            </a>
            <a href="#results" className="nav-link">
              Results
            </a>
            <a href="#about" className="nav-link">
              Notes
            </a>
          </div>
        </nav>

        <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-6 pb-16 lg:px-10">
          <section className="hero-panel">
            <div className="grid gap-10 lg:grid-cols-[1.2fr_0.8fr] lg:items-end">
              <div className="space-y-7">
                <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/6 px-4 py-2 text-sm text-cyan-100/80 backdrop-blur">
                  <Sparkles className="h-4 w-4 text-cyan-300" />
                  Acoustic intelligence for underwater vessel recognition
                </div>

                <div className="space-y-4">
                  <h2 className="max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-5xl lg:text-6xl">
                    A cleaner command deck for marine sound classification.
                  </h2>
                  <p className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg">
                    Upload a hydrophone recording, choose an inference model, and review ranked vessel probabilities in a UI
                    that feels closer to a polished product than a prototype.
                  </p>
                </div>

                <div className="grid gap-4 sm:grid-cols-3">
                  {insightCards.map((card) => {
                    const Icon = card.icon
                    return (
                      <div key={card.label} className="stat-tile">
                        <Icon className="h-5 w-5 text-cyan-300" />
                        <p className="text-xs uppercase tracking-[0.25em] text-slate-400">{card.label}</p>
                        <p className="text-base font-semibold text-white">{card.value}</p>
                      </div>
                    )
                  })}
                </div>
              </div>

              <div className="ocean-card">
                <div className="space-y-5">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/60">Live Status</p>
                      <p className="mt-2 text-2xl font-semibold text-white">Inference Pipeline</p>
                    </div>
                    <div className="status-pill">
                      <span className="status-dot" />
                      Ready
                    </div>
                  </div>

                  <div className="space-y-4" id="workflow">
                    {pipelineSteps.map((step, index) => (
                      <div key={step} className="flex items-start gap-4 rounded-2xl border border-white/8 bg-black/10 p-4">
                        <div className="step-index">{index + 1}</div>
                        <p className="text-sm leading-6 text-slate-200">{step}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="grid gap-8 xl:grid-cols-[1.05fr_0.95fr]">
            <div className="panel-card p-6 sm:p-7">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Control Room</p>
                  <h3 className="section-title">Prepare the analysis</h3>
                </div>
                <div className="icon-shell">
                  <Settings2 className="h-5 w-5 text-cyan-200" />
                </div>
              </div>

              <div className="mt-8 space-y-8">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-200">Choose a model</label>
                    <span className="text-xs uppercase tracking-[0.2em] text-slate-500">4 available</span>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    {modelOptions.map((option) => {
                      const Icon = option.icon
                      const active = model === option.id

                      return (
                        <button
                          key={option.id}
                          onClick={() => setModel(option.id)}
                          className={`model-card ${active ? 'model-card-active' : ''}`}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="space-y-2 text-left">
                              <div className="flex items-center gap-2">
                                <Icon className={`h-4 w-4 ${active ? 'text-cyan-200' : 'text-slate-400'}`} />
                                <p className="font-medium text-white">{option.name}</p>
                              </div>
                              <p className="text-sm leading-6 text-slate-400">{option.description}</p>
                            </div>
                            {active && <CheckCircle2 className="mt-1 h-5 w-5 flex-shrink-0 text-cyan-300" />}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-200">Upload audio</label>
                    <span className="text-xs uppercase tracking-[0.2em] text-slate-500">WAV only</span>
                  </div>

                  <div
                    className={`upload-stage ${dragActive ? 'upload-stage-active' : ''} ${file ? 'upload-stage-filled' : ''}`}
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
                      <div className="flex flex-col items-center gap-3 text-center">
                        <div className="icon-shell !bg-emerald-400/15 !text-emerald-200">
                          <CheckCircle2 className="h-6 w-6" />
                        </div>
                        <div className="space-y-1">
                          <p className="text-base font-semibold text-white">{file.name}</p>
                          <p className="text-sm text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB ready for inference</p>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-4 text-center">
                        <div className="upload-orb">
                          <Upload className="h-7 w-7 text-cyan-200" />
                        </div>
                        <div className="space-y-2">
                          <p className="text-lg font-medium text-white">Drop your acoustic segment here</p>
                          <p className="mx-auto max-w-md text-sm leading-6 text-slate-400">
                            Click to browse or drag a waveform directly into the panel. The classifier expects `.wav`
                            recordings from the ShipsEar preprocessing flow.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <button onClick={handleUpload} disabled={!file || loading} className="action-button">
                  {loading ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Running inference
                    </>
                  ) : (
                    <>
                      Analyze recording
                      <ArrowRight className="h-5 w-5" />
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="space-y-8" id="results">
              {error && (
                <div className="alert-panel">
                  <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-rose-300" />
                  <div>
                    <p className="font-semibold text-rose-100">Classification Error</p>
                    <p className="mt-1 text-sm leading-6 text-rose-100/75">{error}</p>
                  </div>
                </div>
              )}

              {!result && !loading && (
                <div className="panel-card flex min-h-[560px] flex-col justify-between p-6 sm:p-7">
                  <div className="panel-header">
                    <div>
                      <p className="eyebrow">Result Space</p>
                      <h3 className="section-title">Waiting for a recording</h3>
                    </div>
                    <div className="icon-shell">
                      <Ship className="h-5 w-5 text-cyan-200" />
                    </div>
                  </div>

                  <div className="grid gap-6 py-6">
                    <div className="result-placeholder">
                      <div className="radar-ring">
                        <Waves className="h-10 w-10 text-cyan-300" />
                      </div>
                      <div className="space-y-2 text-center">
                        <p className="text-xl font-semibold text-white">No prediction yet</p>
                        <p className="max-w-md text-sm leading-6 text-slate-400">
                          Once you upload a file, the ranked class probabilities, confidence score, and interpretation
                          summary will appear here.
                        </p>
                      </div>
                    </div>

                    <div className="grid gap-4 sm:grid-cols-3">
                      {insightCards.map((card) => (
                        <div key={card.label} className="mini-tile">
                          <p className="text-xs uppercase tracking-[0.25em] text-slate-500">{card.label}</p>
                          <p className="mt-2 text-sm font-medium text-slate-200">{card.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {loading && (
                <div className="panel-card flex min-h-[560px] flex-col items-center justify-center gap-8 p-8 text-center">
                  <div className="loader-shell">
                    <div className="loader-ring" />
                    <div className="loader-core">
                      <Loader2 className="h-10 w-10 animate-spin text-cyan-200" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <p className="eyebrow !text-cyan-200/70">Processing</p>
                    <h3 className="text-3xl font-semibold text-white">Analyzing vessel signature</h3>
                    <p className="max-w-lg text-sm leading-7 text-slate-400">
                      Building features, running the selected classifier, and assembling the confidence ranking for your
                      uploaded acoustic segment.
                    </p>
                  </div>
                </div>
              )}

              {result && (
                <div className="space-y-8">
                  <div className="result-hero">
                    <div className="space-y-6">
                      <div className="flex items-center gap-2 text-cyan-100">
                        <CheckCircle2 className="h-5 w-5 text-cyan-300" />
                        <p className="eyebrow !mb-0 !text-cyan-100/70">Prediction Ready</p>
                      </div>

                      <div className="space-y-3">
                        <p className="text-sm uppercase tracking-[0.3em] text-slate-300/60">Top Classification</p>
                        <h3 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">{result.class_name}</h3>
                        <p className="max-w-xl text-sm leading-7 text-slate-300">{resultSummary}</p>
                      </div>

                      <div className="grid gap-4 sm:grid-cols-3">
                        <div className="result-metric">
                          <p className="text-xs uppercase tracking-[0.22em] text-slate-300/60">Confidence</p>
                          <p className="mt-2 text-3xl font-semibold text-white">{topConfidence}%</p>
                        </div>
                        <div className="result-metric">
                          <p className="text-xs uppercase tracking-[0.22em] text-slate-300/60">Assessment</p>
                          <p className="mt-2 text-lg font-semibold text-white">{resultTone}</p>
                        </div>
                        <div className="result-metric">
                          <p className="text-xs uppercase tracking-[0.22em] text-slate-300/60">Model</p>
                          <p className="mt-2 text-lg font-semibold text-white">{formatModelName(model)}</p>
                        </div>
                      </div>
                    </div>

                    <div className="score-orb">
                      <div className="score-inner">
                        <span className="text-sm uppercase tracking-[0.25em] text-cyan-100/70">Score</span>
                        <span className="text-5xl font-semibold text-white">{topConfidence}</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
                    <div className="panel-card p-6 sm:p-7">
                      <div className="panel-header">
                        <div>
                          <p className="eyebrow">Probability Map</p>
                          <h3 className="section-title">Class ranking</h3>
                        </div>
                        <div className="icon-shell">
                          <BarChart3 className="h-5 w-5 text-cyan-200" />
                        </div>
                      </div>

                      <div className="mt-6 h-[320px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 18, bottom: 8, left: -20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.12)" horizontal={false} />
                            <XAxis type="number" hide domain={[0, 100]} />
                            <YAxis dataKey="name" type="category" stroke="#b9c4d3" fontSize={12} width={120} />
                            <Tooltip
                              cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
                              contentStyle={{
                                backgroundColor: '#082032',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: '14px',
                                color: '#f8fafc',
                              }}
                            />
                            <Bar dataKey="confidence" radius={[0, 8, 8, 0]}>
                              {chartData.map((entry, index) => (
                                <Cell key={`${entry.name}-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="space-y-8">
                      <div className="panel-card p-6 sm:p-7">
                        <div className="panel-header">
                          <div>
                            <p className="eyebrow">Signal Preview</p>
                            <h3 className="section-title">Acoustic pattern</h3>
                          </div>
                          <div className="icon-shell">
                            <AudioWaveform className="h-5 w-5 text-cyan-200" />
                          </div>
                        </div>

                        <div className="signal-panel">
                          <div className="signal-bars">
                            {waveformBars.map((height, index) => (
                              <div
                                key={`${height}-${index}`}
                                className="signal-bar"
                                style={{ height: `${height}%`, animationDelay: `${index * 0.08}s` }}
                              />
                            ))}
                          </div>
                          <p className="text-sm leading-6 text-slate-400">
                            Visual playback is illustrative, but the prediction is based on the uploaded waveform
                            features returned by the backend model.
                          </p>
                        </div>
                      </div>

                      <div className="panel-card p-6 sm:p-7" id="about">
                        <div className="panel-header">
                          <div>
                            <p className="eyebrow">Interpretation</p>
                            <h3 className="section-title">Quick readout</h3>
                          </div>
                          <div className="icon-shell">
                            <Info className="h-5 w-5 text-cyan-200" />
                          </div>
                        </div>

                        <div className="mt-6 space-y-4 text-sm leading-7 text-slate-300">
                          <p>
                            The classifier evaluated the uploaded segment and assigned the highest probability to
                            <span className="font-semibold text-white"> {result.class_name}</span>.
                          </p>
                          <p>
                            Confidence concentration suggests a <span className="font-semibold text-white">{resultTone.toLowerCase()}</span>,
                            which makes this panel useful for both quick demos and side-by-side model comparisons.
                          </p>
                          <div className="rounded-2xl border border-cyan-300/10 bg-cyan-300/5 p-4 text-cyan-50/85">
                            The backend still drives the prediction logic. This refresh focuses only on presentation,
                            clarity, and overall product feel.
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

export default App
