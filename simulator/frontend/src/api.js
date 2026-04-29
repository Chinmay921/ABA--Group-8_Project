/**
 * api.js — thin wrappers around the FastAPI backend.
 * All requests go to /api/* which Vite proxies to http://localhost:8000/api/*.
 */

const BASE = '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}: ${await res.text()}`)
  return res.json()
}

async function post(path, body = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}: ${await res.text()}`)
  return res.json()
}

/** Return which policies are available and the dataset length. */
export const getPolicies = () => get('/policies')

/** Return the list of month labels for the start-period dropdown. */
export const getDates = () => get('/dates')

/**
 * Reset the interactive simulation.
 * @param {string} policy  'manual'|'ppo'|'ddpg'|'taylor'|'random'
 * @param {number} startIdx  index into the monthly dataset
 */
export const resetSim = (policy, startIdx) =>
  post('/reset', { policy, start_idx: startIdx })

/**
 * Advance the interactive simulation by one step.
 * @param {number} action  rate change in [-1, +1] (ignored for non-manual policies)
 */
export const stepSim = (action = 0.0) => post('/step', { action })

/** Retrieve the full trajectory of the current interactive session. */
export const getTrajectory = () => get('/trajectory')

/**
 * Run a complete episode with the given policy (non-interactive).
 * @param {string} policy
 * @param {number} seed
 */
export const runEpisode = (policy, seed = 42) =>
  post('/run-episode', { policy, seed })

/** Run all available policies and return their trajectories for comparison. */
export const compareAll = () => post('/compare')
