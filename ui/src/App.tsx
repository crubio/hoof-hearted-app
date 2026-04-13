import { useQuery } from '@tanstack/react-query'
import TrackList from './components/TrackList'
import type { Track } from './types'
import { fetchTracks } from './api'

export default function App() {
  const { data: tracks, isLoading, isError, error } = useQuery<Track[]>({
    queryKey: ['tracks'],
    queryFn: fetchTracks,
    refetchInterval: 60_000,
  })

  return (
    <div className="container">
      <header>
        <h1>🏇 Hoof Hearted - AI handicapping analysis</h1>
        <h2>Today's races — {new Date().toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' })}</h2>
      </header>
      <main>
        {isLoading && <p>Fetching today's tracks...</p>}
        {isError && <p className="error">Error: {(error as Error).message}</p>}
        {tracks && <TrackList tracks={tracks} />}
      </main>
    </div>
  )
}