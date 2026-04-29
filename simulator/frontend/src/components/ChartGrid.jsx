import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { POLICY_COLORS } from '../App'

// ──────────────────────────────────────────────────────────────────────────────
// Shared chart styling
// ──────────────────────────────────────────────────────────────────────────────

const C = {
  grid:       '#1e2a4a',
  axis:       '#4a5568',
  tooltip:    '#0f1629',
  tooltipBdr: '#1e2a4a',
  target:     '#475569',
}

const TARGET_COLORS = {
  inflation:    '#ff6b6b',
  unemployment: '#ffd166',
  gdp_growth:   '#06d6a0',
  interest_rate: '#4a9eff',
  action:       '#a78bfa',
}

// ──────────────────────────────────────────────────────────────────────────────
// Reusable line chart
// ──────────────────────────────────────────────────────────────────────────────

function MacroChart({ data, dataKey, label, target, unit = '%', domain }) {
  const color   = TARGET_COLORS[dataKey] ?? '#94a3b8'
  const current = data.length > 1 ? data[data.length - 1][dataKey] : null

  // Distance from target → colour-code the header value
  const distColor = current !== null && target !== null
    ? Math.abs(current - target) < 0.5 ? '#00d4aa' : Math.abs(current - target) < 1.5 ? '#ffd166' : '#ff4757'
    : '#e2e8f0'

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span className="chart-title">{label}</span>
        {current !== null && (
          <div className="chart-header-right">
            <span className="chart-current" style={{ color: distColor }}>
              {current.toFixed(2)}{unit}
            </span>
            {target !== null && (
              <span className="chart-target-badge">
                target {target}{unit}
              </span>
            )}
          </div>
        )}
      </div>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={data} margin={{ top: 6, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
          <XAxis
            dataKey="step"
            stroke={C.axis}
            tick={{ fontSize: 10, fill: C.axis }}
            tickLine={false}
          />
          <YAxis
            stroke={C.axis}
            tick={{ fontSize: 10, fill: C.axis }}
            tickLine={false}
            domain={domain ?? ['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              background: C.tooltip,
              border: `1px solid ${C.tooltipBdr}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={v => `Month ${v}`}
            formatter={v => [`${v?.toFixed(3)}${unit}`, label]}
          />
          {target !== null && (
            <ReferenceLine
              y={target}
              stroke={C.target}
              strokeDasharray="5 4"
            />
          )}
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: color }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Rate-action bar-style chart (step function)
// ──────────────────────────────────────────────────────────────────────────────

function ActionChart({ data, policyColor }) {
  return (
    <div className="chart-card">
      <div className="chart-header">
        <span className="chart-title">⚙️ Rate Changes Taken</span>
      </div>
      <ResponsiveContainer width="100%" height={155}>
        <LineChart data={data} margin={{ top: 6, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
          <XAxis dataKey="step" stroke={C.axis} tick={{ fontSize: 10, fill: C.axis }} tickLine={false} />
          <YAxis stroke={C.axis} tick={{ fontSize: 10, fill: C.axis }} tickLine={false} domain={[-1.1, 1.1]} />
          <Tooltip
            contentStyle={{ background: C.tooltip, border: `1px solid ${C.tooltipBdr}`, borderRadius: 8, fontSize: 12 }}
            labelFormatter={v => `Month ${v}`}
            formatter={v => [`${v?.toFixed(3)} pp`, 'Δ Rate']}
          />
          <ReferenceLine y={0} stroke={C.target} strokeDasharray="3 3" />
          <Line type="stepAfter" dataKey="action" stroke={policyColor} strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Cumulative reward chart
// ──────────────────────────────────────────────────────────────────────────────

function RewardChart({ data }) {
  const cumData = data.reduce((acc, pt) => {
    const prev = acc.length > 0 ? acc[acc.length - 1].cum : 0
    return [...acc, { ...pt, cum: prev + (pt.reward ?? 0) }]
  }, [])

  const total = cumData.length > 0 ? cumData[cumData.length - 1].cum : 0

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span className="chart-title">📊 Cumulative Reward</span>
        <div className="chart-header-right">
          <span className="chart-current" style={{ color: '#a78bfa' }}>
            {total.toFixed(1)}
          </span>
          <span className="chart-target-badge">higher = better</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={155}>
        <LineChart data={cumData} margin={{ top: 6, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
          <XAxis dataKey="step" stroke={C.axis} tick={{ fontSize: 10, fill: C.axis }} tickLine={false} />
          <YAxis stroke={C.axis} tick={{ fontSize: 10, fill: C.axis }} tickLine={false} />
          <Tooltip
            contentStyle={{ background: C.tooltip, border: `1px solid ${C.tooltipBdr}`, borderRadius: 8, fontSize: 12 }}
            labelFormatter={v => `Month ${v}`}
            formatter={v => [v?.toFixed(2), 'Cumulative Reward']}
          />
          <ReferenceLine y={0} stroke={C.target} strokeDasharray="3 3" />
          <Line type="monotone" dataKey="cum" stroke="#a78bfa" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Stat card
// ──────────────────────────────────────────────────────────────────────────────

function StatCard({ label, value, target, unit, icon }) {
  if (value == null) return null
  const diff   = target != null ? value - target : null
  const absDiff = diff != null ? Math.abs(diff) : null
  const status = absDiff == null ? 'neutral'
    : absDiff < 0.5 ? 'good'
    : absDiff < 1.5 ? 'warn'
    : 'bad'

  const statusColor = { good: '#00d4aa', warn: '#ffd166', bad: '#ff4757', neutral: '#94a3b8' }[status]

  return (
    <div className="stat-card" style={{ borderColor: `${statusColor}40` }}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <span className="stat-label">{label}</span>
        <span className="stat-value" style={{ color: statusColor }}>
          {value.toFixed(2)}{unit}
        </span>
        {diff != null && (
          <span className="stat-diff" style={{ color: statusColor }}>
            {diff >= 0 ? '+' : ''}{diff.toFixed(2)} from target
          </span>
        )}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main ChartGrid export
// ──────────────────────────────────────────────────────────────────────────────

export default function ChartGrid({ trajectory, targets, currentState, totalReward, policy }) {
  const color = POLICY_COLORS[policy] ?? '#a78bfa'
  const step  = trajectory.length - 1

  const hasData = trajectory.length > 1

  return (
    <main className="chart-area">

      {/* ── Stats row ── */}
      {currentState && (
        <div className="stats-row">
          <StatCard icon="📈" label="Inflation"     value={currentState.inflation}     target={targets.inflation}    unit="%" />
          <StatCard icon="👷" label="Unemployment"  value={currentState.unemployment}  target={targets.unemployment} unit="%" />
          <StatCard icon="💹" label="GDP Growth"    value={currentState.gdp_growth}    target={targets.gdp_growth}   unit="%" />
          <StatCard icon="🏦" label="Interest Rate" value={currentState.interest_rate} target={targets.neutral_rate} unit="%" />
          <div className="stat-card reward-card">
            <div className="stat-icon">🏆</div>
            <div className="stat-content">
              <span className="stat-label">Total Reward</span>
              <span className="stat-value" style={{ color: '#a78bfa' }}>{totalReward.toFixed(1)}</span>
              <span className="stat-diff" style={{ color: '#64748b' }}>Month {step}</span>
            </div>
          </div>
        </div>
      )}

      {/* ── Empty state ── */}
      {!hasData && (
        <div className="empty-state">
          <div className="empty-icon">🏦</div>
          <h2>Ready to Simulate</h2>
          <p>
            Select a policy, choose a starting period, then click <strong>Reset</strong> to initialise the economy.
          </p>
          <div className="empty-hints">
            <div className="hint-chip">🎮 <strong>Manual</strong> — You set the rate each month</div>
            <div className="hint-chip">🤖 <strong>PPO / DDPG</strong> — Trained RL agents auto-run</div>
            <div className="hint-chip">📐 <strong>Taylor Rule</strong> — Classical policy benchmark</div>
            <div className="hint-chip">📊 Switch to <em>Policy Comparison</em> tab to compare all agents</div>
          </div>
        </div>
      )}

      {/* ── Charts ── */}
      {hasData && (
        <>
          <div className="charts-grid-2x2">
            <MacroChart
              data={trajectory}
              dataKey="inflation"
              label="📈 Inflation"
              target={targets.inflation}
              unit="%"
            />
            <MacroChart
              data={trajectory}
              dataKey="unemployment"
              label="👷 Unemployment"
              target={targets.unemployment}
              unit="%"
            />
            <MacroChart
              data={trajectory}
              dataKey="gdp_growth"
              label="💹 GDP Growth"
              target={targets.gdp_growth}
              unit="%"
            />
            <MacroChart
              data={trajectory}
              dataKey="interest_rate"
              label="🏦 Interest Rate"
              target={targets.neutral_rate}
              unit="%"
            />
          </div>

          <div className="charts-row-bottom">
            <ActionChart data={trajectory} policyColor={color} />
            <RewardChart data={trajectory} />
          </div>
        </>
      )}
    </main>
  )
}
