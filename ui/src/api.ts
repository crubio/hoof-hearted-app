import axios from 'axios'
import type { Track, AnalysisResult } from './types'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const api = axios.create({ baseURL: BASE_URL })

export async function fetchTracks(): Promise<Track[]> {
  const { data } = await api.get('/tracks')
  return data
}

export async function fetchProgram(trackId: string, raceN: number): Promise<void> {
  await api.get(`/program/${trackId}/${raceN}`)
}

export async function fetchAnalysis(trackId: string, raceN: number): Promise<AnalysisResult> {
  const { data } = await api.post<AnalysisResult>(`/analyze/${trackId}/${raceN}`)
  return data
}