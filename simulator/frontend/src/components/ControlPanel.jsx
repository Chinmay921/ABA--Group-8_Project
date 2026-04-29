import { POLICY_LABELS, POLICY_COLORS } from '../App'

/**
 * ControlPanel — left sidebar with policy selector, period picker,
 * manual rate slider, speed controls, and action buttons.
 */
export default function ControlPanel({
  policy, setPolicy,
  availPolicies,
  startIdx, setStartIdx,
  dates,
  manualAction, setManualAction,
  speed, setSpeed,
  onReset, onStep, onAutoRun,
  isRunning, isDone, isInit,
}) {
  const enabledPolicies = Object.entries(availPolicies)
    .filter(([, avail]) => avail)
    .map(([k]) => k)

  const actionColor =
    manualAction > 0.05 ? '#ff6b6b' :
    manualAction < -0.05 ? '#00d4aa' : '#94a3b8'

  const actionLabel =
    manualAction > 0.05 ? 'Rate Hike ↑' :
    manualAction < -0.05 ? 'Rate Cut ↓' : 'Hold Steady'

  return (
    <aside className="control-panel">

      {/* ── Policy Selector ── */}
      <section className="panel-section">
        <h3 className="section-label">Policy</h3>
        <div className="policy-grid">
          {enabledPolicies.map(p => (
            <button
              key={p}
              className={`policy-btn ${policy === p ? 'selected' : ''}`}
              style={policy === p
                ? { borderColor: POLICY_COLORS[p], color: POLICY_COLORS[p], background: `${POLICY_COLORS[p]}15` }
                : {}}
              onClick={() => setPolicy(p)}
            >
              {POLICY_LABELS[p] || p}
            </button>
          ))}
        </div>

        {/* Policy description */}
        <div className="policy-desc">
          {policy === 'manual'  && 'You act as the Fed Chair — set the interest rate change each month.'}
          {policy === 'ppo'     && 'Pre-trained PPO agent (500k timesteps). Learns by trial and error.'}
          {policy === 'ddpg'    && 'Pre-trained DDPG agent (500k timesteps). Deterministic actor-critic.'}
          {policy === 'taylor'  && 'Classical Taylor Rule: reacts to inflation and GDP growth gaps.'}
          {policy === 'random'  && 'Uniform random rate changes in [−1, +1]. Lower-bound baseline.'}
        </div>
      </section>

      {/* ── Starting Period ── */}
      <section className="panel-section">
        <h3 className="section-label">Starting Period</h3>
        <select
          className="select-input"
          value={startIdx}
          onChange={e => setStartIdx(Number(e.target.value))}
        >
          {dates.length > 0
            ? dates.map((d, i) => <option key={i} value={i}>{d}</option>)
            : <option value={0}>Full dataset</option>
          }
        </select>
        <div className="hint-text">
          Tip: start from 2020-01 to test COVID stress response.
        </div>
      </section>

      {/* ── Manual Rate Slider (Manual mode only) ── */}
      {policy === 'manual' && (
        <section className="panel-section">
          <h3 className="section-label">Rate Change</h3>
          <div className="slider-group">
            <div className="slider-value-row">
              <span className="slider-value" style={{ color: actionColor }}>
                {manualAction >= 0 ? '+' : ''}{manualAction.toFixed(2)} pp
              </span>
              <span className="action-label" style={{ color: actionColor }}>
                {actionLabel}
              </span>
            </div>

            <input
              type="range"
              min="-1" max="1" step="0.05"
              value={manualAction}
              onChange={e => setManualAction(Number(e.target.value))}
              className="slider"
              style={{ accentColor: actionColor }}
            />

            <div className="slider-labels">
              <span>−1.0 pp</span>
              <span style={{ color: '#64748b' }}>Hold</span>
              <span>+1.0 pp</span>
            </div>
          </div>
        </section>
      )}

      {/* ── Speed Control (Agent modes) ── */}
      {policy !== 'manual' && (
        <section className="panel-section">
          <h3 className="section-label">Playback Speed</h3>
          <div className="speed-group">
            {[['🐢 Slow', 600], ['▶ Normal', 200], ['⚡ Fast', 50]].map(([label, ms]) => (
              <button
                key={ms}
                className={`speed-btn ${speed === ms ? 'active' : ''}`}
                onClick={() => setSpeed(ms)}
              >
                {label}
              </button>
            ))}
          </div>
        </section>
      )}

      {/* ── Action Buttons ── */}
      <section className="panel-section controls">
        <button className="btn btn-reset" onClick={onReset}>
          🔄 Reset
        </button>

        {policy === 'manual' ? (
          <button
            className="btn btn-step"
            onClick={onStep}
            disabled={!isInit || isDone}
          >
            ▶ Step (1 Month)
          </button>
        ) : (
          <button
            className={`btn ${isRunning ? 'btn-pause' : isDone ? 'btn-done' : 'btn-run'}`}
            onClick={onAutoRun}
            disabled={(!isInit && !isRunning) || (isDone && !isRunning)}
          >
            {isRunning ? '⏸ Pause' : isDone ? '✅ Done' : '▶ Auto Run'}
          </button>
        )}
      </section>

      {/* ── Done badge ── */}
      {isDone && (
        <div className="done-badge">
          Episode complete — click Reset to restart.
        </div>
      )}

      {/* ── Targets reference ── */}
      <div className="targets-card">
        <div className="targets-title">Policy Targets</div>
        <div className="target-row">
          <span>Inflation</span>
          <span className="target-val">2.0 %</span>
        </div>
        <div className="target-row">
          <span>Unemployment</span>
          <span className="target-val">4.5 %</span>
        </div>
        <div className="target-row">
          <span>GDP Growth</span>
          <span className="target-val">0.25 %/mo</span>
        </div>
        <div className="target-row">
          <span>Neutral Rate</span>
          <span className="target-val">2.5 %</span>
        </div>
      </div>

      {/* ── Reward formula ── */}
      <div className="formula-card">
        <div className="targets-title">Reward Function</div>
        <code className="formula">
          −( (π−2)² + 0.5(u−4.5)² + 0.5(g−0.25)² + 0.1(Δr)² )
        </code>
        <div className="hint-text">
          Reaches 0 only when all targets are met simultaneously.
        </div>
      </div>

    </aside>
  )
}
