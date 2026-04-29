import { useState, useEffect, useRef } from 'react'
import * as api from './api'
import ControlPanel from './components/ControlPanel'
import ChartGrid from './components/ChartGrid'
import CompareView from './components/CompareView'

// ──────────────────────────────────────────────────────────────────────────────
// Constants shared across components
// ──────────────────────────────────────────────────────────────────────────────

export const POLICY_LABELS = {
  manual: '🎮 Manual Control',
  ppo:    '🤖 PPO Agent',
  ddpg:   '🧠 DDPG Agent',
  taylor: '📐 Taylor Rule',
  random: '🎲 Random Policy',
}

export const POLICY_COLORS = {
  manual: '#a78bfa',
  ppo:    '#00d4aa',
  ddpg:   '#ff9500',
  taylor: '#4a9eff',
  random: '#ff4757',
}

const DEFAULT_TARGETS = {
  inflation:    2.0,
  unemployment: 4.5,
  gdp_growth:   0.25,
  neutral_rate: 2.5,
}

// ──────────────────────────────────────────────────────────────────────────────
// App
// ──────────────────────────────────────────────────────────────────────────────

export default function App() {
  // Tabs
  const [tab, setTab] = useState('interactive')

  // Backend metadata
  const [availPolicies, setAvailPolicies] = useState({
    manual: true, random: true, taylor: true, ppo: false, ddpg: false,
  })
  const [dates, setDates]   = useState([])
  const [backendOk, setBackendOk] = useState(null)  // null=checking, true, false

  // Simulation state
  const [policy,       setPolicy]       = useState('manual')
  const [startIdx,     setStartIdx]     = useState(0)
  const [trajectory,   setTrajectory]   = useState([])
  const [targets,      setTargets]      = useState(DEFAULT_TARGETS)
  const [isRunning,    setIsRunning]    = useState(false)
  const [isDone,       setIsDone]       = useState(false)
  const [isInit,       setIsInit]       = useState(false)
  const [manualAction, setManualAction] = useState(0.0)
  const [speed,        setSpeed]        = useState(200)   // ms between auto-steps
  const [errorMsg,     setErrorMsg]     = useState(null)

  // Compare tab state
  const [compareResults, setCompareResults] = useState(null)
  const [isComparing,    setIsComparing]    = useState(false)

  const runnerRef    = useRef(null)
  const isRunningRef = useRef(false)

  // ── Startup: check backend ────────────────────────────────────────────────
  useEffect(() => {
    api.getPolicies()
      .then(data => {
        setAvailPolicies(data.available)
        setBackendOk(true)
      })
      .catch(() => setBackendOk(false))

    api.getDates()
      .then(data => setDates(data.dates || []))
      .catch(() => {})
  }, [])

  // ── Cleanup timer on unmount ──────────────────────────────────────────────
  useEffect(() => () => { if (runnerRef.current) clearInterval(runnerRef.current) }, [])

  // ── Helpers ───────────────────────────────────────────────────────────────

  const stopRunner = () => {
    if (runnerRef.current) { clearInterval(runnerRef.current); runnerRef.current = null }
    setIsRunning(false)
    isRunningRef.current = false
  }

  // ── Interactive handlers ──────────────────────────────────────────────────

  const handleReset = async () => {
    stopRunner()
    setErrorMsg(null)
    try {
      const result = await api.resetSim(policy, startIdx)
      setTrajectory([result.state])
      setTargets({ ...DEFAULT_TARGETS, ...result.targets })
      setIsDone(false)
      setIsInit(true)
    } catch (e) {
      setErrorMsg(`Reset failed: ${e.message}`)
    }
  }

  const handleStep = async () => {
    if (!isInit || isDone) return
    setErrorMsg(null)
    try {
      const result = await api.stepSim(manualAction)
      setTrajectory(prev => [...prev, result])
      if (result.done) setIsDone(true)
    } catch (e) {
      setErrorMsg(`Step failed: ${e.message}`)
    }
  }

  const handleAutoRun = () => {
    if (isRunning) { stopRunner(); return }
    if (!isInit || isDone) return

    setIsRunning(true)
    isRunningRef.current = true

    const tick = async () => {
      if (!isRunningRef.current) return
      try {
        // For non-manual policies the backend ignores the action value
        const result = await api.stepSim(policy === 'manual' ? manualAction : 0.0)
        setTrajectory(prev => [...prev, result])
        if (result.done) {
          stopRunner()
          setIsDone(true)
        }
      } catch (e) {
        stopRunner()
        setErrorMsg(`Auto-run error: ${e.message}`)
      }
    }

    runnerRef.current = setInterval(tick, speed)
  }

  // ── Comparison handler ────────────────────────────────────────────────────

  const handleCompare = async () => {
    setIsComparing(true)
    setCompareResults(null)
    setErrorMsg(null)
    try {
      const results = await api.compareAll()
      setCompareResults(results)
    } catch (e) {
      setErrorMsg(`Comparison failed: ${e.message}`)
    }
    setIsComparing(false)
  }

  // ── Derived values ────────────────────────────────────────────────────────

  const currentState = trajectory[trajectory.length - 1] ?? null
  const totalReward  = trajectory.reduce((s, p) => s + (p.reward ?? 0), 0)

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-left">
          <span className="header-icon">🏦</span>
          <div>
            <h1>Monetary Policy Simulator</h1>
            <p>ABA Group 8 — DTU Business Analytics · RL-based Central Banking</p>
          </div>
        </div>

        <nav className="tab-nav">
          <button
            className={`tab-btn ${tab === 'interactive' ? 'active' : ''}`}
            onClick={() => setTab('interactive')}
          >
            🎮 Interactive Simulator
          </button>
          <button
            className={`tab-btn ${tab === 'compare' ? 'active' : ''}`}
            onClick={() => setTab('compare')}
          >
            📊 Policy Comparison
          </button>
        </nav>

        {/* Backend status indicator */}
        <div className="backend-status">
          {backendOk === null && <span className="status-dot checking" />}
          {backendOk === true  && <span className="status-dot online" title="Backend connected" />}
          {backendOk === false && (
            <span className="status-dot offline" title="Backend offline — run: uvicorn api:app --reload --port 8000" />
          )}
          <span className="status-label">
            {backendOk === null  ? 'Connecting…' :
             backendOk === true  ? 'API Online' :
                                   'API Offline'}
          </span>
        </div>
      </header>

      {/* ── Error banner ── */}
      {errorMsg && (
        <div className="error-banner">
          ⚠️ {errorMsg}
          <button onClick={() => setErrorMsg(null)}>✕</button>
        </div>
      )}

      {/* ── Tab content ── */}
      {tab === 'interactive' && (
        <div className="interactive-layout">
          <ControlPanel
            policy={policy}           setPolicy={setPolicy}
            availPolicies={availPolicies}
            startIdx={startIdx}       setStartIdx={setStartIdx}
            dates={dates}
            manualAction={manualAction} setManualAction={setManualAction}
            speed={speed}             setSpeed={setSpeed}
            onReset={handleReset}
            onStep={handleStep}
            onAutoRun={handleAutoRun}
            isRunning={isRunning}
            isDone={isDone}
            isInit={isInit}
          />
          <ChartGrid
            trajectory={trajectory}
            targets={targets}
            currentState={currentState}
            totalReward={totalReward}
            policy={policy}
          />
        </div>
      )}

      {tab === 'compare' && (
        <CompareView
          onCompare={handleCompare}
          isComparing={isComparing}
          results={compareResults}
          availPolicies={availPolicies}
        />
      )}
    </div>
  )
}
