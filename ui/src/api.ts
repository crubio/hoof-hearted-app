import axios from 'axios'
import type { Track, AnalysisResult } from './types'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const api = axios.create({ baseURL: BASE_URL })

export async function fetchTracks(): Promise<Track[]> {
  const { data } = await api.get('/tracks')
  return data
}

export async function fetchAnalysis(trackId: string, raceN: number): Promise<AnalysisResult> {
  try {
    const { data } = await api.post<AnalysisResult>(`/analyze/${trackId}/${raceN}`)
    return data
  } catch (err) {
    // The backend now returns a real 4xx/5xx with {"error": "..."} on failure (rate limit,
    // auth, upstream error, etc.) instead of a 200 with a blank result -- surface that
    // message instead of axios's generic "Request failed with status code 429".
    if (axios.isAxiosError(err)) {
      const message = (err.response?.data as { error?: string } | undefined)?.error
      if (message) throw new Error(message)
    }
    throw err
  }
}
