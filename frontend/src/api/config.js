// In development and GitHub Codespaces, relative URL '' routes all /api requests
// through the Vite dev server proxy (http://127.0.0.1:8000), eliminating cross-origin
// CORS preflights (OPTIONS) and port-binding discrepancies across environments.
// If an external production API URL is explicitly configured, respect it.
const configuredBase = import.meta.env.VITE_API_BASE_URL

export const API_BASE_URL = (
  configuredBase && !configuredBase.includes('localhost')
    ? configuredBase
    : ''
).replace(/\/+$/, '')
