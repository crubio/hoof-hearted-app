import { useQuery } from '@tanstack/react-query'
import TrackList from './components/TrackList'
import type { Track } from './types'
import { fetchTracks } from './api'

export default function App() {
  // isPending (no data yet) implies isLoading in TanStack Query v5 -- isLoading is redundant here.
  const { data: tracks, isError, error, isPending } = useQuery<Track[]>({
    queryKey: ['tracks'],
    queryFn: fetchTracks,
    refetchInterval: 60_000,
  })

  const noRacesToday = tracks && tracks.length === 0

  return (
    <div className="container">
      <header>
        <h1>🏇 Hoof Hearted - AI handicapping analysis</h1>
        <h2>Today's races — {new Date().toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' })}</h2>
      </header>
      <main>
        {isPending && <p>Fetching today's tracks...</p>}
        {isError && <p className="error">Error: {(error as Error).message}</p>}
        {noRacesToday && (
          <div className="terminal-alert terminal-alert-warning">
            No races today. Check back later.
          </div>
        )}
        {tracks && <TrackList tracks={tracks} />}
      </main>
    </div>
  )
}