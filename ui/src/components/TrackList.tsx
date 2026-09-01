import type { Track } from '../types'
import TrackRow from './TrackRow'

interface Props {
  tracks: Track[]
}

// Track filtering happens server-side (app/scraper.py FILTERED_TRACKS) -- no separate
// client-side allowlist here, so the two can't drift out of sync with each other.
export default function TrackList({ tracks }: Props) {
  if (tracks.length === 0) {
    return <p>No tracked races available today.</p>
  }

  const allClosed = tracks.every(t => t.status === 'closed' || t.status === 'Closed')

  return (
    <div className="track-list">
      {allClosed && (
        <div className="terminal-alert terminal-alert-warning">
          All tracks closed right now. Check back later.
        </div>
      )}
      {tracks.map(track => (
        <TrackRow key={track.brisCode} track={track} />
      ))}
    </div>
  )
}
