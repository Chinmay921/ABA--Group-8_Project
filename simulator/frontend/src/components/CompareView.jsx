import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts'
import { POLICY_LABELS, POLICY_COLORS } from '../App'

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const C = {
  grid:   '#1e2a4a',
  axis:   '#4a5568',
  tt:     '#0f1629',
  ttBdr:  '#1e2a4a',
}

const TARGETS = {
  inflation:    2.0,
  unemployment: 4.5,
  gdp_growth:   0.25,
  interest_rate: 2.5,
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-policy line chart
// ──────────────────────────────────────────────────────────────────────────────

function CompareChart({ results, dataKey, label, unit = '%' }) {
  const policies = Object.keys(results)
  const maxLen   = Math.max(...policies.map(p => results[p].trajectory.length))
  const target   = TARGETS[dataKey] ?? null

  const data = Array.from({ length: maxLen }, (_, i) => {
    const pt = { step: i }
    policies.forEach(p => {
      const t = results[p].trajectory
      if (t[i]) pt[p] = t[i][dataKey]
    })
    return pt
  })

  return (
    <div className="compare-chart-card">
      <div className="chart-header">
        <span className="chart-title">{label}</span>
        {target != null && (
          <span className="chart-target-badge">target {target}{unit}</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={data} margin={{ top: 6, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
          <XAxis dataKey="step" stroke={C.axis} tick={{ fontSize: 10, fill: C.axis }} tickLine={false} />
          <YAxis stroke={C.axis} tick={{ fontSize: 10, fill: C.axis }} tickLine={false} />
          <Tooltip
            contentStyle={{ background: C.tt, border: `1px solid ${C.ttBdr}`, borderRadius: 8, fontSize: 12 }}
            labelFormatter={v => `Month ${v}`}
            formatter={(v, name) => [`${v?.toFixed(3)}${unit}`, POLICY_LABELS[name] ?? name]}
          />
          {target != null && (
            <ReferenceLine y={target} stroke="#475569" strokeDasharray="5 4" />
          )}
          <Legend
            formatter={val => (
              <span style={{ color: POLICY_COLORS[val] ?? '#94a3b8', fontSize: 12 }}>
                {POLICY_LABELS[val] ?? val}
              </span>
            )}
          />
          {policies.map(p => (
            <Line
              key={p}
              type="monotone"
              dataKey={p}
              stroke={POLICY_COLORS[p] ?? '#94a3b8'}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Summary table
// ──────────────────────────────────────────────────────────────────────────────

function SummaryTable({ results }) {
  const sorted = Object.entries(results).sort((a, b) => b[1].total_reward - a[1].total_reward)

  const rewards   = sorted.map(([, d]) => d.total_reward)
  const maxReward = Math.max(...rewards)
  const minReward = Math.min(...rewards)
  const range     = maxReward - minReward || 1

  return (
    <div className="summary-table-wrapper">
      <h3 className="section-label" style={{ marginBottom: 16 }}>Results Summary</h3>
      <table className="summary-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Policy</th>
            <th>Total Reward</th>
            <th>vs. Taylor Rule</th>
            <th>Steps</th>
            <th>Performance</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([pol, data], idx) => {
            const taylorReward = results['taylor']?.total_reward ?? null
            const vsTaylor = taylorReward != null && pol !== 'taylor'
              ? data.total_reward - taylorReward
              : null
            const barPct = ((data.total_reward - minReward) / range) * 100

            return (
              <tr key={pol} className={idx === 0 ? 'best-row' : ''}>
                <td className="rank-cell">{idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : idx + 1}</td>
                <td>
                  <div className="policy-name-cell">
                    <span className="policy-dot" style={{ background: POLICY_COLORS[pol] }} />
                    <span style={{ color: POLICY_COLORS[pol], fontWeight: 600 }}>
                      {POLICY_LABELS[pol] ?? pol}
                    </span>
                  </div>
                </td>
                <td className="reward-cell" style={{ color: POLICY_COLORS[pol] }}>
                  {data.total_reward.toFixed(1)}
                </td>
                <td className="vs-cell">
                  {vsTaylor != null ? (
                    <span style={{ color: vsTaylor > 0 ? '#00d4aa' : '#ff4757' }}>
                      {vsTaylor >= 0 ? '+' : ''}{vsTaylor.toFixed(1)}
                    </span>
                  ) : '—'}
                </td>
                <td style={{ color: '#64748b' }}>{data.steps}</td>
                <td>
                  <div className="perf-bar-track">
                    <div
                      className="perf-bar-fill"
                      style={{ width: `${barPct}%`, background: POLICY_COLORS[pol] }}
                    />
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <div className="table-note">
        All rewards are negative — <strong>closer to 0 is better</strong>.
        Reward = −(inflation_gap² + 0.5×unemployment_gap² + 0.5×GDP_gap² + 0.1×rate_change²).
        "vs. Taylor Rule" shows how many reward points the RL agent gains over the classical benchmark.
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// CompareView export
// ──────────────────────────────────────────────────────────────────────────────

export default function CompareView({ onCompare, isComparing, results, availPolicies }) {
  return (
    <div className="compare-view">

      {/* ── Header ── */}
      <div className="compare-header">
        <div>
          <h2>Policy Comparison</h2>
          <p>
            Runs <strong>PPO</strong>, <strong>DDPG</strong>, and <strong>Taylor Rule</strong> over
            the <em>same</em> full 95-step episode and compares their macroeconomic
            trajectories and total reward against the classical benchmark.
          </p>
        </div>
        <button className="btn btn-primary btn-large" onClick={onCompare} disabled={isComparing}>
          {isComparing ? '⏳ Running simulations…' : '▶ Run All Policies'}
        </button>
      </div>

      {/* ── Loading ── */}
      {isComparing && (
        <div className="loading-state">
          <div className="spinner" />
          <p>Simulating policies… this takes a few seconds.</p>
        </div>
      )}

      {/* ── Results ── */}
      {results && !isComparing && (
        <>
          <SummaryTable results={results} />

          <div className="compare-charts-grid">
            <CompareChart results={results} dataKey="inflation"     label="📈 Inflation"     unit="%" />
            <CompareChart results={results} dataKey="unemployment"  label="👷 Unemployment"  unit="%" />
            <CompareChart results={results} dataKey="gdp_growth"    label="💹 GDP Growth"    unit="%" />
            <CompareChart results={results} dataKey="interest_rate" label="🏦 Interest Rate" unit="%" />
          </div>
        </>
      )}

      {/* ── Empty state ── */}
      {!results && !isComparing && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h2>Compare All Policies</h2>
          <p>
            Click "Run All Policies" to run PPO, DDPG, and Taylor Rule in the same environment
            and visualise their macroeconomic trajectories side-by-side.
          </p>

          <div className="policy-legend-preview">
            {['ppo', 'ddpg', 'taylor'].map(key => (
              <div key={key} className="legend-preview-item" style={{ opacity: availPolicies[key] ? 1 : 0.4 }}>
                <span className="policy-dot" style={{ background: POLICY_COLORS[key] }} />
                <span style={{ color: POLICY_COLORS[key] }}>{POLICY_LABELS[key]}</span>
                {!availPolicies[key] && (
                  <span style={{ fontSize: 10, color: '#64748b', marginLeft: 4 }}>(model not loaded)</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
