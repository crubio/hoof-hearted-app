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
  success: boolean
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