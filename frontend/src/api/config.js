const isCodespaceBrowser =
  typeof window !== 'undefined' &&
  (window.location.hostname.endsWith('.github.dev') ||
    window.location.hostname.endsWith('.app.github.dev'))

const defaultApiBaseUrl = isCodespaceBrowser ? '' : 'http://localhost:8000'

export const API_BASE_URL = (
  isCodespaceBrowser &&
  (!import.meta.env.VITE_API_BASE_URL ||
    import.meta.env.VITE_API_BASE_URL.includes('localhost'))
    ? ''
    : import.meta.env.VITE_API_BASE_URL || defaultApiBaseUrl
).replace(/\/+$/, '')
