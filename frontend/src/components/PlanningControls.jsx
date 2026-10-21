import { Sparkles, RotateCcw, ArrowRight, Loader2, Compass, AlertCircle } from 'lucide-react'
import LocationInput from './LocationInput'
import TravelPreferences from './TravelPreferences'

export default function PlanningControls({
  origin,
  setOrigin,
  destination,
  setDestination,
  selectionTarget,
  setSelectionTarget,
  onReset,
  presetRoutes,
  onApplyPreset,
  selectedTimestamp,
  setSelectedTimestamp,
  trafficWeight,
  setTrafficWeight,
  trafficInfo,
  onFindAndCompare,
  onCalculateShortestOnly,
  isCalculating,
  calculatingMode,
  routeError,
  onGoToElectronicCity,
  clearRouteRequest,
  setRoute,
  setComparison,
  setManualCameraAction,
}) {
  return (
    <div className="space-y-3.5">
      {/* Locations Card */}
      <div className="rounded-xl border border-slate-200 bg-white p-3.5 space-y-3">
        <div className="flex items-center justify-between pb-1 border-b border-slate-100">
          <h2 className="text-xs font-semibold text-slate-800">Route Endpoints</h2>
          <button
            type="button"
            onClick={onReset}
            disabled={!origin && !destination}
            className="flex items-center gap-1 text-[11px] font-medium text-slate-500 hover:text-slate-800 disabled:opacity-30 disabled:pointer-events-none transition-colors"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Reset</span>
          </button>
        </div>

        <LocationInput
          id="starting-location"
          label="Starting Location (A)"
          variant="origin"
          placeholder="Search place, landmark, or click map..."
          value={origin}
          onChange={(loc) => {
            clearRouteRequest()
            setRoute(null)
            setComparison(null)
            setManualCameraAction(null)
            setOrigin(loc)
          }}
          onClear={() => {
            clearRouteRequest()
            setRoute(null)
            setComparison(null)
            setManualCameraAction(null)
            setOrigin(null)
          }}
          isActiveTarget={selectionTarget === 'origin'}
          onActivateTarget={() => setSelectionTarget('origin')}
        />

        <LocationInput
          id="destination-location"
          label="Destination (B)"
          variant="destination"
          placeholder="Search place, landmark, or click map..."
          value={destination}
          onChange={(loc) => {
            clearRouteRequest()
            setRoute(null)
            setComparison(null)
            setManualCameraAction(null)
            setDestination(loc)
          }}
          onClear={() => {
            clearRouteRequest()
            setRoute(null)
            setComparison(null)
            setManualCameraAction(null)
            setDestination(null)
          }}
          isActiveTarget={selectionTarget === 'destination'}
          onActivateTarget={() => setSelectionTarget('destination')}
        />

        {/* Quick Sample Routes */}
        <div className="pt-1">
          <div className="flex items-center gap-1 text-[11px] font-medium text-slate-500 mb-1.5">
            <Sparkles className="h-3 w-3 text-amber-500" />
            <span>Quick Electronic City sample routes:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {presetRoutes.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => onApplyPreset(preset)}
                className="rounded-lg border border-slate-200 bg-slate-50/70 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:border-blue-300 hover:bg-blue-50/50 hover:text-blue-700 transition-all text-left"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Travel Preferences Card */}
      <TravelPreferences
        selectedTimestamp={selectedTimestamp}
        onTimestampChange={(val) => {
          clearRouteRequest()
          setComparison(null)
          setSelectedTimestamp(val)
        }}
        trafficWeight={trafficWeight}
        onTrafficWeightChange={(val) => {
          clearRouteRequest()
          setComparison(null)
          setTrafficWeight(val)
        }}
        presets={trafficInfo?.sample_presets}
      />

      {/* Error Notice */}
      {routeError && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50/80 p-3 text-xs text-rose-800">
          <AlertCircle className="h-4 w-4 text-rose-600 mt-0.5 shrink-0" />
          <div className="space-y-0.5">
            <strong className="font-semibold">Routing Notice:</strong>
            <p className="text-[11px] leading-relaxed text-rose-700">{routeError}</p>
          </div>
        </div>
      )}

      {/* Primary & Secondary Action Buttons */}
      <div className="space-y-2 pt-0.5">
        <button
          type="button"
          onClick={onFindAndCompare}
          disabled={!origin || !destination || isCalculating}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white shadow-xs hover:bg-blue-700 active:scale-[0.99] disabled:opacity-40 disabled:pointer-events-none transition-all cursor-pointer"
        >
          {isCalculating && calculatingMode === 'compare' ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin-fast" />
              <span>Finding & comparing routes…</span>
            </>
          ) : (
            <>
              <span>Find and compare routes</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </>
          )}
        </button>

        <button
          type="button"
          onClick={onCalculateShortestOnly}
          disabled={!origin || !destination || isCalculating}
          className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none transition-all cursor-pointer"
        >
          {isCalculating && calculatingMode === 'shortest' ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin-fast text-slate-500" />
              <span>Calculating shortest route…</span>
            </>
          ) : (
            <span>Calculate shortest route only</span>
          )}
        </button>
      </div>

      {/* Unobtrusive Coverage Info & Quick Go to Electronic City action */}
      <div className="flex items-center justify-between rounded-xl border border-slate-200/80 bg-slate-50/60 px-3 py-2 text-[11px] text-slate-500">
        <span>Dataset covers 602 road sections in Electronic City</span>
        {onGoToElectronicCity && (
          <button
            type="button"
            onClick={onGoToElectronicCity}
            className="flex items-center gap-1 font-medium text-blue-600 hover:text-blue-800"
          >
            <Compass className="h-3 w-3" />
            <span>Focus area</span>
          </button>
        )}
      </div>
    </div>
  )
}
