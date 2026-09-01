import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import { fetchAnalysis } from '../api'
import type { Track } from '../types'

interface Props {
  track: Track
}

export default function TrackRow({ track }: Props) {
  const trackId = track.brisCode.toLowerCase()
  // track.races is typed as required, but the upstream payload is passed through largely
  // unvalidated -- default to {} so a track missing it degrades to "no races" instead of
  // throwing out of Object.values and blanking the whole app.
  const raceList = Object.values(track.races ?? {}).sort((a, b) => a.raceNumber - b.raceNumber)

  // All hooks are called unconditionally, every render -- any early return lives below them.
  const [selectedRaceN, setSelectedRaceN] = useState(track.currentRaceNumber)
  const [expanded, setExpanded] = useState(true)
  // The analyzed text is stored alongside the race number it describes. Without that pairing,
  // a poll that advances track.currentRaceNumber (or the user picking a different race) would
  // keep showing the previous race's stale selections under the new race's header.
  const [result, setResult] = useState<{ raceN: number; text: string } | null>(null)

  const selectedRace = raceList.find(r => r.raceNumber === selectedRaceN) ?? raceList[0]

  const { mutate: analyze, isPending, isError, error } = useMutation({
    // POST /analyze/{track}/{race} already fetches+caches the program on a cache miss, so
    // there's no need to call /program first -- that was a fully redundant round trip.
    mutationFn: () => fetchAnalysis(trackId, selectedRaceN),
    onSuccess: (data) => {
      setResult({ raceN: selectedRaceN, text: data.analysis })
      setExpanded(true)
    },
  })

  if (!selectedRace) return null

  const postTime = selectedRace.postTime
    ? new Date(selectedRace.postTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'TBD'

  const analysis = result && result.raceN === selectedRaceN ? result.text : null

  const statusBadge = () => {
    if (isPending) return <span className="badge badge--pending">⏳ Analyzing...</span>
    if (isError)   return <span className="badge badge--error">✗ Failed</span>
    if (analysis)  return <span className="badge badge--success">✓ Analyzed</span>
    return null
  }

  return (
    <div className="track-row">
      <div className="track-header">
        <div className="track-meta">
          <span className="track-name">{track.name}</span>
          {raceList.length > 1 ? (
            <select
              className="race-picker"
              value={selectedRaceN}
              onChange={e => setSelectedRaceN(Number(e.target.value))}
              aria-label="Race number"
            >
              {raceList.map(r => (
                <option key={r.raceNumber} value={r.raceNumber}>
                  Race {r.raceNumber}{r.raceNumber === track.currentRaceNumber ? ' (current)' : ''}
                </option>
              ))}
            </select>
          ) : (
            <span className="track-detail">Race {selectedRaceN}</span>
          )}
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
        <p className="pending"><strong>Running AI analysis — free-tier models can take 20–40 seconds...</strong></p>
      )}

      {analysis && expanded && (
        <div className="analysis-output">
          <ReactMarkdown>{analysis}</ReactMarkdown>
        </div>
      )}
    </div>
  )
}
