import { apiRequest } from './apiClient'

export function calculateShortestRoute(origin, destination, signal) {
  return apiRequest('/api/routes/shortest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ origin, destination }),
    signal,
  })
}
