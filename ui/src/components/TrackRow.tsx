import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import { fetchProgram, fetchAnalysis } from '../api'
import type { Track } from '../types'

interface Props {
  track: Track
}

export default function TrackRow({ track }: Props) {
  const [analysis, setAnalysis] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(true)

  const trackId = track.brisCode.toLowerCase()
  const raceN = track.currentRaceNumber

  // dig into races to find the current race for post time + meta
  const currentRace = Object.values(track.races).find(
    r => r.raceNumber === raceN
  ) ?? Object.values(track.races)[0]

  if (!currentRace) return null

  const postTime = currentRace?.postTime
    ? new Date(currentRace.postTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'TBD'

  const { mutate: analyze, isPending, isError, error, isSuccess } = useMutation({
    mutationFn: async () => {
      await fetchProgram(trackId, raceN)
      return fetchAnalysis(trackId, raceN)
    },
    onSuccess: (data) => {
      setAnalysis(data.analysis)
      setExpanded(true)}
  })

  const statusBadge = () => {
    if (isPending) return <span className="badge badge--pending">⏳ Analyzing...</span>
    if (isError)   return <span className="badge badge--error">✗ Failed</span>
    if (isSuccess) return <span className="badge badge--success">✓ Analyzed</span>
    return null
  }

  return (
    <div className="track-row">
      <div className="track-header">
        <div className="track-meta">
          <span className="track-name">{track.name}</span>
          <span className="track-detail">Race {raceN}</span>
          <span className="track-detail">Post {postTime}</span>
          <span className="track-status">{track.status}</span>
          {statusBadge()}
        </div>
        <div className="track-actions">
          {analysis && (
            <button onClick={() => setExpanded(e => !e)}>
              {expanded ? '▲ Collapse' : '▼ Expand'}
            </button>
          )}
          <button onClick={() => analyze()} disabled={isPending}>
            {isPending ? 'Analyzing...' : analysis ? 'Re-analyze' : 'Analyze'}
          </button>

        </div>
      </div>

      {isError && (
        <p className="error">Analysis failed: {(error as Error).message}</p>
      )}

      {isPending && (
        <p className="pending">⏳ Running AI analysis — this takes 5–10 seconds...</p>
      )}

      {analysis && expanded && (
        <div className="analysis-output">
          <ReactMarkdown>{analysis}</ReactMarkdown>
        </div>
      )}
    </div>
  )
}