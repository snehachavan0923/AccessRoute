import { API_BASE_URL } from '../api/config'

export async function apiRequest(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)

  if (!response.ok) {
    const errorBody = await response.json()
    throw new Error(
      errorBody.detail || `API request failed with status ${response.status}`,
    )
  }

  return response.json()
}
