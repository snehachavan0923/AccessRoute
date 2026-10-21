import { apiRequest } from './apiClient'

export function calculateShortestRoute(origin, destination, signal) {
  return apiRequest('/api/routes/shortest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ origin, destination }),
    signal,
  })
}

export function compareRoutes(origin, destination, options = {}, signal) {
  const { timestamp, timeIndex, trafficWeight } = options
  return apiRequest('/api/routes/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      origin,
      destination,
      timestamp: timestamp || undefined,
      time_index:
        timeIndex !== undefined && timeIndex !== '' ? Number(timeIndex) : undefined,
      traffic_weight:
        trafficWeight !== undefined ? Number(trafficWeight) : 2.0,
    }),
    signal,
  })
}

export function getTrafficInfo(signal) {
  return apiRequest('/api/routes/traffic-info', {
    method: 'GET',
    signal,
  })
}

export function searchPlaces(query, signal) {
  const encoded = encodeURIComponent(query.trim())
  return apiRequest(`/api/geocoding/search?q=${encoded}`, {
    method: 'GET',
    signal,
  })
}
