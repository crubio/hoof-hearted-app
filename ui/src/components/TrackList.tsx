import type { Track } from '../types'
import TrackRow from './TrackRow'

const ALLOWED_TRACKS = ['sa', 'kee', 'aqu', 'op', 'cd']

interface Props {
  tracks: Track[]
}

export default function TrackList({ tracks }: Props) {
  const filtered = tracks.filter(t =>
    ALLOWED_TRACKS.includes(t.brisCode?.toLowerCase())
  )

  if (filtered.length === 0) {
    return <p>No tracked races available today.</p>
  }

  const allClosed = filtered.every(t => t.status === 'closed' || t.status === 'Closed')

  return (
    <div className="track-list">
      {allClosed && (
        <p>All tracked closed right now. Check back later</p>
      )}
      {filtered.map(track => (
        <TrackRow key={track.brisCode} track={track} />
      ))}
    </div>
  )
}