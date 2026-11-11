import { useEffect, useState } from 'react'
import './App.css'

const tabs = ['overview', 'analyze', 'results', 'reports']

const formatNumber = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—'
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 3,
  })
}

const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return `${(Number(value) * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`
}

function AttributionLine({ values }) {
  if (!values?.length) return null
  const width = 720
  const height = 150
  const peak = Math.max(...values, 1e-8)
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width
      const y = height - (value / peak) * (height - 12) - 6
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  return (
    <svg className="attribution-line" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Integrated Gradients attribution over the EEG window">
      <line x1="0" y1={height - 1} x2={width} y2={height - 1} />
      <polyline points={points} />
    </svg>
  )
}

function AttributionHeatmap({ data }) {
  if (!data?.importance?.length) return null
  const rows = data.importance
  const width = 720
  const height = 240
  const cellWidth = width / Math.max(rows[0]?.length || 1, 1)
  const cellHeight = height / Math.max(rows.length, 1)
  return (
    <svg className="attribution-heatmap" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Integrated Gradients time-frequency attribution map">
      {rows.flatMap((row, y) => row.map((value, x) => (
        <rect
          key={`${x}-${y}`}
          x={x * cellWidth}
          y={height - (y + 1) * cellHeight}
          width={cellWidth + 0.25}
          height={cellHeight + 0.25}
          fill={`hsl(${220 - Math.min(Math.max(value, 0), 1) * 200} 90% 52%)`}
          opacity={0.18 + Math.min(Math.max(value, 0), 1) * 0.82}
        />
      ))) }
    </svg>
  )
}

function MetricTrendChart({ title, caption, points, field, color, maxValue, formatter }) {
  if (!points?.length) return <div className="metric-chart-empty">Evaluation data is not available.</div>
  const width = 520
  const height = 210
  const left = 42
  const right = 18
  const top = 18
  const bottom = 42
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const values = points.map((point) => Number(point[field]) || 0)
  const ceiling = maxValue ?? Math.max(...values, 0.1) * 1.12
  const xy = points.map((point, index) => ({
    x: left + (points.length === 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth),
    y: top + plotHeight - (values[index] / ceiling) * plotHeight,
    value: values[index],
    horizon: point.horizon_minutes,
  }))
  return (
    <section className="metric-chart-card">
      <div className="chart-card-heading"><div><h3>{title}</h3><p>{caption}</p></div><span className="chart-key" style={{ '--chart-color': color }} /></div>
      <svg viewBox={`0 0 ${width} ${height}`} className="metric-trend-chart" role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = top + plotHeight - ratio * plotHeight
          return <g key={ratio}><line x1={left} y1={y} x2={width - right} y2={y} className="chart-gridline"/><text x={left - 8} y={y + 3} textAnchor="end" className="chart-tick">{formatter(ratio * ceiling)}</text></g>
        })}
        <polyline points={xy.map((point) => `${point.x},${point.y}`).join(' ')} fill="none" stroke={color} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        {xy.map((point) => <g key={point.horizon}><circle cx={point.x} cy={point.y} r="4.5" fill={color} stroke="#0e1722" strokeWidth="2"/><text x={point.x} y={height - 19} textAnchor="middle" className="chart-tick">{point.horizon} min</text><text x={point.x} y={point.y - 11} textAnchor="middle" className="chart-value">{formatter(point.value)}</text></g>)}
      </svg>
    </section>
  )
}

function PredictionCurve({ points, threshold }) {
  if (!points?.length) return null
  const width = 960
  const height = 230
  const left = 48
  const right = 14
  const top = 16
  const bottom = 34
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const firstTime = Number(points[0].time_minutes) || 0
  const lastTime = Number(points.at(-1).time_minutes) || firstTime + 1
  const xy = points.map((point) => ({
    x: left + ((Number(point.time_minutes) - firstTime) / Math.max(lastTime - firstTime, 1e-8)) * plotWidth,
    y: top + plotHeight - Math.max(0, Math.min(1, Number(point.preictal_probability) || 0)) * plotHeight,
  }))
  const thresholdY = top + plotHeight - threshold * plotHeight
  return (
    <div className="prediction-chart-wrap"><svg className="prediction-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Preictal prediction score across the complete EDF recording">
      {[0, 0.25, 0.5, 0.75, 1].map((value) => { const y = top + plotHeight - value * plotHeight; return <g key={value}><line x1={left} y1={y} x2={width - right} y2={y} className="chart-gridline"/><text x={left - 9} y={y + 3} textAnchor="end" className="chart-tick">{value.toFixed(2)}</text></g> })}
      <line x1={left} y1={thresholdY} x2={width - right} y2={thresholdY} className="threshold-line"/>
      <polyline points={xy.map((point) => `${point.x},${point.y}`).join(' ')} className="risk-line"/>
      <text x={left} y={height - 10} className="chart-tick">{firstTime.toFixed(1)} min</text><text x={width - right} y={height - 10} textAnchor="end" className="chart-tick">{lastTime.toFixed(1)} min</text>
      <text x={width - right - 4} y={thresholdY - 6} textAnchor="end" className="threshold-label">Threshold {threshold.toFixed(2)}</text>
    </svg></div>
  )
}

function EEGTrace({ trace }) {
  if (!trace?.channels?.length) return null
  const width = 960
  const left = 88
  const right = 14
  const top = 12
  const laneHeight = 45
  const height = top + trace.channels.length * laneHeight + 27
  const plotWidth = width - left - right
  const colors = ['#48d8bc', '#69aefb', '#f2bd72', '#d493f5']
  return (
    <div className="prediction-chart-wrap"><svg className="signal-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Actual preprocessed EEG channels from the highest-scoring prediction window">
      {trace.channels.map((channel, index) => {
        const laneTop = top + index * laneHeight
        const values = channel.values || []
        const maxAbs = Math.max(1, ...values.map((value) => Math.abs(value)))
        const path = values.map((value, sample) => {
          const x = left + (sample / Math.max(values.length - 1, 1)) * plotWidth
          const y = laneTop + laneHeight / 2 - Math.max(-1, Math.min(1, value / maxAbs)) * (laneHeight * 0.38)
          return `${sample ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`
        }).join(' ')
        return <g key={channel.channel}><line x1={left} y1={laneTop + laneHeight / 2} x2={width - right} y2={laneTop + laneHeight / 2} className="chart-gridline"/><text x={left - 10} y={laneTop + laneHeight / 2 + 3} textAnchor="end" className="channel-label">{channel.channel}</text><path d={path} fill="none" stroke={colors[index % colors.length]} strokeWidth="1.25" vectorEffect="non-scaling-stroke"/></g>
      })}
      <text x={left} y={height - 5} className="chart-tick">0 s</text><text x={left + plotWidth / 2} y={height - 5} textAnchor="middle" className="chart-tick">{(trace.window_seconds / 2).toFixed(0)} s</text><text x={width - right} y={height - 5} textAnchor="end" className="chart-tick">{trace.window_seconds.toFixed(0)} s</text>
    </svg></div>
  )
}

function SignalSpectrum({ data }) {
  if (!data?.relative_power?.length) return null
  const width = 600
  const height = 190
  const left = 38
  const right = 12
  const top = 12
  const bottom = 30
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const maxFrequency = Math.max(...data.frequencies_hz, 1)
  const points = data.relative_power.map((power, index) => {
    const x = left + (data.frequencies_hz[index] / maxFrequency) * plotWidth
    const y = top + (1 - power) * plotHeight
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return <svg className="domain-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Relative EEG power spectrum across channels"><line x1={left} y1={top + plotHeight} x2={width - right} y2={top + plotHeight} className="chart-gridline"/><line x1={left} y1={top} x2={left} y2={top + plotHeight} className="chart-gridline"/><polyline points={points} className="spectrum-line"/><text x={left} y={height - 8} className="chart-tick">0 Hz</text><text x={width - right} y={height - 8} textAnchor="end" className="chart-tick">45 Hz</text><text x={9} y={top + 5} className="chart-tick">high</text><text x={13} y={top + plotHeight} className="chart-tick">low</text></svg>
}

function SignalSpectrogram({ data }) {
  if (!data?.relative_power?.length) return null
  const rows = data.relative_power
  const width = 600
  const height = 190
  const left = 42
  const right = 12
  const top = 12
  const bottom = 30
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const cellWidth = plotWidth / Math.max(rows[0]?.length || 1, 1)
  const cellHeight = plotHeight / Math.max(rows.length, 1)
  return <svg className="domain-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Short-time Fourier time-frequency representation of the EEG window"><rect x={left} y={top} width={plotWidth} height={plotHeight} fill="#0b1420"/>{rows.flatMap((row, frequency) => row.map((power, time) => <rect key={`${frequency}-${time}`} x={left + time * cellWidth} y={top + plotHeight - (frequency + 1) * cellHeight} width={cellWidth + .25} height={cellHeight + .25} fill={`hsl(${220 - power * 200} 88% ${20 + power * 42}%)`}/>))}<text x={left} y={height - 8} className="chart-tick">0 s</text><text x={width - right} y={height - 8} textAnchor="end" className="chart-tick">{data.times_seconds.at(-1)?.toFixed(1) || '10'} s</text><text x={8} y={top + 5} className="chart-tick">45 Hz</text><text x={18} y={top + plotHeight} className="chart-tick">0</text></svg>
}

function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [overview, setOverview] = useState({ total_rows: 0, case_count: 0, people_count: 0, rows: [], best_model: null })
  const [reportText, setReportText] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [datasetRecordings, setDatasetRecordings] = useState([])
  const [datasetRecording, setDatasetRecording] = useState('')
  const [explain, setExplain] = useState(false)
  const [includeShap, setIncludeShap] = useState(false)
  const [horizonMinutes, setHorizonMinutes] = useState(30)
  const [scoreThreshold, setScoreThreshold] = useState(0.5)
  const [analysis, setAnalysis] = useState(null)
  const [loadingOverview, setLoadingOverview] = useState(true)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadOverview = async () => {
      try {
        const response = await fetch('/api/overview')
        const payload = await response.json()
        setOverview(payload)
      } catch (err) {
        setError('Could not load the research summary from the backend.')
      } finally {
        setLoadingOverview(false)
      }
    }

    const loadReports = async () => {
      try {
        const response = await fetch('/api/reports')
        const payload = await response.json()
        setReportText(payload.report_text || 'No report is available yet.')
      } catch (err) {
        setError('Could not load the final report preview.')
      }
    }

    const loadRecordings = async () => {
      try {
        const response = await fetch('/api/recordings')
        const payload = await response.json()
        const items = payload.recordings || []
        setDatasetRecordings(items)
        if (items.length) setDatasetRecording(items[0].path)
      } catch (err) {
        setDatasetRecordings([])
      }
    }

    loadOverview()
    loadReports()
    loadRecordings()
  }, [])

  const handleAnalyze = async (event) => {
    event.preventDefault()
    if (!selectedFile && !datasetRecording) {
      setError('Choose a downloaded CHB-MIT recording or upload an EDF file.')
      return
    }

    const formData = new FormData()
    if (selectedFile) formData.append('file', selectedFile)
    setLoadingAnalysis(true)
    setError('')

    try {
      const query = new URLSearchParams({
        explain: String(explain),
        include_shap: String(explain && includeShap),
        horizon_minutes: String(horizonMinutes),
        score_threshold: String(scoreThreshold),
        ...(selectedFile ? {} : { recording: datasetRecording }),
      })
      const response = await fetch(`/api/analyze-edf?${query}`, {
        method: 'POST',
        body: formData,
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Analysis failed.')
      }
      setAnalysis(payload)
      setActiveTab('analyze')
    } catch (err) {
      setError(err.message || 'Unexpected analysis error.')
    } finally {
      setLoadingAnalysis(false)
    }
  }

  const primaryCurveRows = (overview.rows || [])
    .filter((row) => row.model_variant === 'multi_horizon' && Number(row.false_alarm_budget_per_hour) === 0.5)
    .sort((a, b) => Number(a.horizon_minutes) - Number(b.horizon_minutes))

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark">E<span>+</span></div>
          <div><strong>NeuroSignal</strong><small>EEG research suite</small></div>
        </div>
        <div className="side-label">WORKSPACE</div>
        <nav className="tab-bar" aria-label="Main sections">
          {tabs.map((tab, index) => (
            <button
              key={tab}
              type="button"
              className={activeTab === tab ? 'tab-button active' : 'tab-button'}
              onClick={() => setActiveTab(tab)}
            >
              <span className="nav-icon">{['◫', '⌁', '▤', '◷'][index]}</span>
              <span>{tab[0].toUpperCase() + tab.slice(1)}</span>
              {tab === 'results' && <span className="nav-count">{overview.total_rows || '—'}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className="status-dot" /> Local research environment
          <small>CHB-MIT · {overview.case_count || '—'} patient cases</small>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">RESEARCH WORKSPACE <span>/</span> {activeTab.toUpperCase()}</p>
            <h1>{activeTab === 'overview' ? 'Seizure forecasting' : activeTab === 'analyze' ? 'EEG analysis' : activeTab === 'results' ? 'Evaluation results' : 'Research report'}</h1>
          </div>
          <div className="header-actions"><span className="connection-pill"><span className="status-dot" /> Model ready</span><span className="user-avatar">NS</span></div>
        </header>

        {error && <div className="alert">{error}</div>}

      {activeTab === 'overview' && (
        <main className="overview-page">
          <section className="welcome-banner">
            <div className="welcome-copy">
              <span className="section-kicker"><span className="status-dot" /> MULTI-HORIZON MODEL · READY</span>
              <h2>Understand seizure risk<br/><span>across prediction horizons.</span></h2>
              <p>Explore patient-held-out results or analyze a CHB-MIT EDF recording with the trained EEG model.</p>
              <button className="primary-action" type="button" onClick={() => setActiveTab('analyze')}>Analyze a recording <span>↗</span></button>
            </div>
            <div className="signal-art" aria-hidden="true">
              <div className="signal-orbit orbit-one"/><div className="signal-orbit orbit-two"/>
              <svg viewBox="0 0 400 170"><path d="M4 93h43l12-4 9 7 10-4 12-39 11 79 13-92 13 83 12-30 12 3 10-4 9 4 12-3 11 8 11-5 10 3 11-5 10 4 12-1 12-4 11 6 12-3 11 2 10-6 10 4 10-2 13 2 10-4 9 5 12-3 10 1 12-4 11 3 12-5 10 5 12-2 14 1 12-3 11 3h33"/></svg>
              <div className="signal-caption"><span>EEG SIGNAL</span><strong>256 <small>Hz</small></strong></div>
            </div>
            <div className="banner-horizon"><span>ACTIVE HORIZONS</span><div>05 <small>MIN</small><i/>10 <small>MIN</small><i/>15 <small>MIN</small><i/>30 <small>MIN</small></div></div>
          </section>

          <div className="section-heading"><div><span className="section-kicker">MODEL PERFORMANCE</span><h2>Evaluation at a glance</h2></div><button className="text-action" onClick={() => setActiveTab('results')}>View all results <span>→</span></button></div>
          <section className="stat-grid dashboard-stats">
              <div className="stat-box">
                <span className="label">LOSO evaluations</span>
                <strong>{loadingOverview ? '…' : formatNumber(overview.total_rows)}</strong>
                <small>Across model and horizon settings</small>
              </div>
              <div className="stat-box">
                <span className="label">Event sensitivity · best row</span>
                <strong>
                  {loadingOverview
                    ? '…'
                    : overview.best_model
                      ? formatPercent(overview.best_model.event_sensitivity_mean)
                      : '—'}
                </strong>
                <small>At a validation FAR budget; observed test FAR can differ</small>
              </div>
              <div className="stat-box">
                <span className="label">Best-performing variant</span>
                <strong className="stat-text">{loadingOverview ? '…' : (overview.best_model?.model_variant || '—').replaceAll('_', ' ')}</strong>
                <small>Selected by event sensitivity</small>
              </div>
              <div className="stat-box">
                <span className="label">Best horizon</span>
                <strong>{loadingOverview ? '…' : overview.best_model?.horizon_minutes ? `${overview.best_model.horizon_minutes} min` : '—'}</strong>
                <small>Selected SOP setting</small>
              </div>
          </section>

          <section className="performance-charts">
            <div className="chart-section-title"><div><span className="section-kicker">PATIENT-HELD-OUT RESULTS</span><h2>How performance changes by horizon</h2></div><span>Primary model · 0.5 FA/h validation budget</span></div>
            <div className="trend-grid">
              <MetricTrendChart title="Event sensitivity" caption="Mean seizure events forecast on held-out patients" points={primaryCurveRows} field="event_sensitivity_mean" color="#42d5bb" maxValue={1} formatter={formatPercent} />
              <MetricTrendChart title="Observed false alarms" caption="Test-set false alarms per eligible interictal hour" points={primaryCurveRows} field="false_alarms_per_hour_mean" color="#70adff" formatter={(value) => `${Number(value).toFixed(2)}`} />
            </div>
            <p className="chart-footnote">The 0.5 FA/h value is a validation threshold-selection budget; observed test-set rates can differ. These are retrospective research results.</p>
          </section>

          <section className="overview-lower">
            <div className="panel feature-panel">
              <div className="panel-title-row"><div><span className="section-kicker">ANALYSIS PIPELINE</span><h2>From signal to insight</h2></div><span className="mini-badge">3 DOMAINS</span></div>
              <div className="pipeline-steps">
                <div><span className="step-number">01</span><div><strong>Preprocess</strong><small>Filter · montage · window</small></div></div><span className="step-arrow">→</span>
                <div><span className="step-number">02</span><div><strong>Fuse features</strong><small>Temporal · spectral · time-frequency</small></div></div><span className="step-arrow">→</span>
                <div><span className="step-number">03</span><div><strong>Forecast</strong><small>Four seizure prediction horizons</small></div></div>
              </div>
            </div>
            <div className="panel notes-panel">
              <span className="section-kicker">STUDY SCOPE</span><h2>Patient-independent evaluation</h2>
              <p>LOSO splits hold out each included patient. CNN, LSTM, CNN-LSTM, and feature ablations are compared.</p>
              <div className="scope-footer"><span>CHB-MIT</span><span>{overview.case_count || '—'} CASES · {overview.people_count || '—'} PEOPLE</span><span>RESEARCH USE</span></div>
            </div>
          </section>
        </main>
      )}

      {activeTab === 'analyze' && (
        <main className="panel">
          <h2>Analyze an EDF recording</h2>
          <form className="upload-form" onSubmit={handleAnalyze}>
            <label className="dataset-picker">
              <span>Downloaded CHB-MIT recording</span>
              <select value={datasetRecording} onChange={(event) => { setDatasetRecording(event.target.value); setSelectedFile(null) }}>
                <option value="">Choose a recording…</option>
                {datasetRecordings.map((item) => <option key={item.path} value={item.path}>{item.patient} · {item.filename}</option>)}
              </select>
              <small>{datasetRecordings.length ? `${datasetRecordings.length} EDF recordings available locally` : 'No local EDF files found in the project data folder'}</small>
            </label>
            <span className="source-divider">OR UPLOAD</span>
            <label className="file-picker">
              <input type="file" accept=".edf" onChange={(event) => { setSelectedFile(event.target.files?.[0] ?? null); if (event.target.files?.[0]) setDatasetRecording('') }} />
              <span>{selectedFile ? selectedFile.name : 'Browse for another EDF file'}</span>
            </label>
            <label className="explain-toggle">
              <input type="checkbox" checked={explain} onChange={(event) => setExplain(event.target.checked)} />
              Explain highest-scoring window with Integrated Gradients
            </label>
            <label className="explain-toggle">
              <span>Prediction SOP</span>
              <select value={horizonMinutes} onChange={(event) => setHorizonMinutes(Number(event.target.value))}>
                {[5, 10, 15, 30].map((minutes) => <option key={minutes} value={minutes}>{minutes} minutes</option>)}
              </select>
            </label>
            <label className="explain-toggle">
              <span>Score threshold</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={scoreThreshold}
                onChange={(event) => setScoreThreshold(Number(event.target.value))}
              />
            </label>
            {explain && (
              <label className="explain-toggle">
                <input type="checkbox" checked={includeShap} onChange={(event) => setIncludeShap(event.target.checked)} />
                Also calculate SHAP (slower; install optional explain dependencies)
              </label>
            )}
            <button type="submit" disabled={(!selectedFile && !datasetRecording) || loadingAnalysis}>
              {loadingAnalysis ? 'Running analysis…' : 'Analyze EDF'}
            </button>
          </form>
          <p className="muted">
            The displayed score is retrospective and uncalibrated. Changing this threshold changes the displayed binary labels; it does not guarantee a false-alarm rate on new patients.
          </p>

          {analysis && (
            <div className="analysis-grid">
              <section className="panel nested">
                <h3>Summary</h3>
                <dl className="metrics-list">
                  <div><dt>Filename</dt><dd>{analysis.summary.filename}</dd></div>
                  <div><dt>Windows</dt><dd>{analysis.summary.window_count}</dd></div>
                  <div><dt>Channels</dt><dd>{analysis.summary.channel_count}</dd></div>
                  <div><dt>Selected SOP</dt><dd>{analysis.summary.prediction_horizon_minutes} min</dd></div>
                  <div><dt>Score threshold</dt><dd>{formatNumber(analysis.summary.score_threshold)}</dd></div>
                  <div><dt>Mean preictal score</dt><dd>{formatNumber(analysis.summary.mean_preictal_probability)}</dd></div>
                  <div><dt>Peak preictal score</dt><dd>{formatNumber(analysis.summary.max_preictal_probability)}</dd></div>
                  <div><dt>Interictal predictions</dt><dd>{analysis.summary.prediction_counts.Interictal}</dd></div>
                  <div><dt>Preictal predictions</dt><dd>{analysis.summary.prediction_counts.Preictal}</dd></div>
                </dl>
              </section>

              <section className="panel nested wide chart-panel">
                <div className="panel-title-row"><div><span className="section-kicker">MODEL OUTPUT · ALL RECORDING</span><h3>Preictal risk over time</h3></div><span className="mini-badge">SOP {analysis.selected_horizon_minutes} MIN</span></div>
                <p className="muted">Each point is the trained model’s score for a 10-second EEG window. The dashed line is the selected display threshold.</p>
                <PredictionCurve points={analysis.plot_data} threshold={analysis.score_threshold} />
                <div className="chart-legend"><span><i className="legend-risk"/>Preictal probability</span><span><i className="legend-threshold"/>Display threshold</span></div>
              </section>

              <section className="panel nested wide chart-panel">
                <div className="panel-title-row"><div><span className="section-kicker">PREPROCESSED EEG · PEAK-SCORE WINDOW</span><h3>Multichannel signal trace</h3></div><span className="mini-badge">{formatNumber(analysis.eeg_trace.window_start_minutes)} MIN</span></div>
                <p className="muted">Actual preprocessed signal from the highest-scoring window. Traces are shown as per-channel z-scores for visual comparison.</p>
                <EEGTrace trace={analysis.eeg_trace} />
              </section>

              <section className="panel nested wide chart-panel">
                <div className="panel-title-row"><div><span className="section-kicker">MULTI-DOMAIN SIGNAL VIEWS · PEAK-SCORE WINDOW</span><h3>Frequency and time-frequency transforms</h3></div><span className="mini-badge">0–45 HZ</span></div>
                <p className="muted">Derived from the selected preprocessed EEG window. Relative scale is for visual comparison and is not a clinical biomarker.</p>
                <div className="transform-grid"><div><h4>Mean relative power spectrum</h4><SignalSpectrum data={analysis.signal_transforms?.spectrum}/></div><div><h4>Short-time Fourier representation</h4><SignalSpectrogram data={analysis.signal_transforms?.spectrogram}/></div></div>
              </section>

              <section className="panel nested">
                <h3>Matched channels</h3>
                <ul className="chip-list">
                  {analysis.matched_channels.map((channel) => (
                    <li key={channel}>{channel}</li>
                  ))}
                </ul>
              </section>

              <section className="panel nested wide">
                <h3>Top predicted windows</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Time (min)</th>
                        <th>Prediction</th>
                        <th>Preictal</th>
                        <th>Interictal</th>
                        <th>State</th>
                        {Object.keys(analysis.predictions[0] || {})
                          .filter((key) => key.startsWith('preictal_probability_sop_'))
                          .map((key) => <th key={key}>{key.replace('preictal_probability_sop_', 'SOP ' )}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.top_predictions.map((row) => (
                        <tr key={`${row.time_minutes}-${row.preictal_probability}`}>
                          <td>{formatNumber(row.time_minutes)}</td>
                          <td>{row.prediction}</td>
                          <td>{formatNumber(row.preictal_probability)}</td>
                          <td>{formatNumber(row.interictal_probability)}</td>
                          <td>{row.predicted_state}</td>
                          {Object.keys(analysis.predictions[0] || {})
                            .filter((key) => key.startsWith('preictal_probability_sop_'))
                            .map((key) => <td key={key}>{formatNumber(row[key])}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
              {analysis.explanation && (
                <section className="panel nested wide">
                  <h3>EEG explanation for the highest-scoring window</h3>
                  <p className="muted">
                    {analysis.explanation.method} channel attribution at {formatNumber(analysis.explanation.window_time_minutes)} min.
                    These are model attributions, not a validated seizure focus.
                  </p>
                  <div className="chip-list">
                    {analysis.explanation.channel_importance
                      .slice()
                      .sort((a, b) => b.importance - a.importance)
                      .slice(0, 12)
                      .map((item) => <span className="chip" key={item.channel}>{item.channel}: {formatNumber(item.importance)}</span>)}
                  </div>
                  <h4>Fusion attention weights</h4>
                  <div className="chip-list">
                    {Object.entries(analysis.explanation.attention_by_domain || {}).map(([domain, weight]) => (
                      <span className="chip" key={domain}>{domain}: {formatNumber(weight)}</span>
                    ))}
                  </div>
                  <h4>Temporal attribution across the window (relative)</h4>
                  <AttributionLine values={analysis.explanation.time_importance} />
                  <div className="chart-axis-labels"><span>0 s</span><span>{analysis.explanation.time_importance_seconds?.at(-1)?.toFixed(1) || '10'} s</span></div>
                  <h4>Time-frequency attribution (0–45 Hz)</h4>
                  <AttributionHeatmap data={analysis.explanation.time_frequency} />
                  {analysis.explanation.shap_channel_importance && (
                    <>
                      <h4>SHAP channel attribution</h4>
                      <div className="chip-list">
                        {analysis.explanation.shap_channel_importance
                          .slice()
                          .sort((a, b) => b.importance - a.importance)
                          .slice(0, 12)
                          .map((item) => <span className="chip" key={item.channel}>{item.channel}: {formatNumber(item.importance)}</span>)}
                      </div>
                    </>
                  )}
                </section>
              )}
            </div>
          )}
        </main>
      )}

      {activeTab === 'results' && (
        <main className="panel">
          <h2>LOSO summary table</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Horizon</th>
                  <th>Target FAR/hour</th>
                  <th>Observed FAR/hour</th>
                  <th>Event sensitivity</th>
                  <th>Window F1</th>
                  <th>People meeting FAR</th>
                </tr>
              </thead>
              <tbody>
                {(overview.rows || []).map((row, index) => (
                  <tr key={`${row.model_variant}-${row.horizon_minutes}-${index}`}>
                    <td>{row.model_variant}</td>
                    <td>{row.horizon_minutes}</td>
                    <td>{formatNumber(row.false_alarm_budget_per_hour)}</td>
                    <td>{formatNumber(row.false_alarms_per_hour_mean)}</td>
                    <td>{formatPercent(row.event_sensitivity_mean)}</td>
                    <td>{formatPercent(row.window_f1_mean)}</td>
                    <td>{row.people_meeting_far_target ?? '—'} / {row.people ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </main>
      )}

      {activeTab === 'reports' && (
        <main className="panel">
          <h2>Final report</h2>
          <pre className="report-box">{reportText}</pre>
        </main>
      )}
      </section>
    </div>
  )
}

export default App
