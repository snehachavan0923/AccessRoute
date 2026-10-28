import { API_BASE_URL } from '../api/config'
import { apiRequest } from './apiClient'

export function getVisionStatus(signal) {
  return apiRequest('/api/vision/status', {
    method: 'GET',
    signal,
  })
}

export function analyzeVideo(payload, signal) {
  return apiRequest('/api/vision/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_id: payload.videoId || 'sample_traffic_video',
      max_frames:
        payload.maxFrames !== undefined && payload.maxFrames !== null && payload.maxFrames !== ''
          ? Number(payload.maxFrames)
          : undefined,
      conf: payload.conf !== undefined ? Number(payload.conf) : 0.5,
      iou: payload.iou !== undefined ? Number(payload.iou) : 0.3,
      min_track_lifespan_frames:
        payload.minTrackLifespanFrames !== undefined
          ? Number(payload.minTrackLifespanFrames)
          : 5,
      generate_annotated_video: payload.generateAnnotatedVideo ?? true,
    }),
    signal,
  })
}

export async function uploadVideo(file, signal) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/vision/upload`, {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}))
    throw new Error(
      errorBody.detail || `Upload failed with status ${response.status}`,
    )
  }

  return response.json()
}

export function getFullMediaUrl(relativePath) {
  if (!relativePath) return ''
  if (relativePath.startsWith('http://') || relativePath.startsWith('https://')) {
    return relativePath
  }
  return `${API_BASE_URL}${relativePath.startsWith('/') ? '' : '/'}${relativePath}`
}
