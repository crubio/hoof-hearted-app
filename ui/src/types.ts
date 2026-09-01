export interface Race {
  raceNumber: number
  raceDate: string
  raceType: string
  purse: string
  raceName: string
  raceDescription: string
  displayRaceDescription: string
  distance: string
  distanceLong: string
  surface: string
  surfaceLabel: string
  ageRestrictions: string
  sexRestrictions: string
  postTime: string | null
  postTimeStamp: number
  mtp: number
  status: string
  grade: string
  maxClaimPrice: string
  wagers: string
  country: string
  formattedPurse: string
  currentRace: boolean
  hasBrisPick: boolean
  hasExpertPick: boolean
  hasEasyBets: boolean
  carryover: unknown[]
}

export interface Track {
  name: string
  brisCode: string
  trackName?: string
  currentRaceNumber: number
  races: Record<string, Race>  // keyed by string index "0", "1", etc.
  status?: string
}

export interface AnalysisResult {
  success: true
  meta: {
    track: string
    race: number
    cache_hit: boolean
    model: string
    tokens: {
      prompt: number
      completion: number
      total: number
    }
    elapsed_ms: number
  }
  analysis: string
}

// Shape returned by POST /analyze/{track}/{race} on failure (rate limit, auth, upstream
// error, etc.) alongside a real 4xx/5xx status -- api.ts converts this into a thrown Error
// before it reaches a component, but the shape is documented here for reference.
export interface AnalysisErrorResponse {
  success: false
  error: string
}