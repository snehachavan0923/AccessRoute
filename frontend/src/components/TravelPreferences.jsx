/* eslint-disable no-unused-vars */
import { Clock, Sliders } from 'lucide-react'

const TIME_PRESETS = [
  { id: 'morning', label: 'Morning Peak — 08:30 AM', timestamp: '2026-01-01 08:30:00' },
  { id: 'afternoon', label: 'Afternoon Off-Peak — 01:00 PM', timestamp: '2026-01-01 13:00:00' },
  { id: 'evening', label: 'Evening Peak — 06:30 PM', timestamp: '2026-01-01 18:30:00' },
  { id: 'night', label: 'Night / Free-Flow — 02:00 AM', timestamp: '2026-01-01 02:00:00' },
]

export default function TravelPreferences({
  selectedTimestamp,
  onTimestampChange,
  trafficWeight,
  onTrafficWeightChange,
  accessibilityMode = 'standard',
  onAccessibilityModeChange,
  presets = TIME_PRESETS,
}) {
  // Map numeric weight to the 3 required descriptions
  function getWeightDescription(weight) {
    if (weight <= 1.0) return 'Prefer shorter distance'
    if (weight <= 3.0) return 'Balanced'
    return 'Avoid more congestion'
  }

  const currentDescription = getWeightDescription(trafficWeight)

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3.5 space-y-3.5">
      <div className="flex items-center gap-1.5 pb-1 border-b border-slate-100">
        <Sliders className="h-4 w-4 text-slate-500" />
        <h3 className="text-xs font-semibold text-slate-800">Travel preferences</h3>
      </div>


      {/* When are you travelling? */}
      <div>
        <label
          htmlFor="traffic-timestamp-select"
          className="flex items-center gap-1.5 text-xs font-medium text-slate-700 mb-1.5"
        >
          <Clock className="h-3.5 w-3.5 text-slate-400" />
          <span>When are you travelling?</span>
        </label>
        <select
          id="traffic-timestamp-select"
          value={selectedTimestamp}
          onChange={(e) => onTimestampChange(e.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-slate-50/70 py-1.5 px-2.5 text-xs font-medium text-slate-800 transition-colors focus:border-blue-500 focus:bg-white focus:outline-hidden focus:ring-2 focus:ring-blue-100"
        >
          {presets.map((preset) => (
            <option key={preset.id} value={preset.timestamp}>
              {preset.label}
            </option>
          ))}
        </select>
      </div>

      {/* How much do you want to avoid traffic? */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <label htmlFor="traffic-weight-slider" className="font-medium text-slate-700">
            How much do you want to avoid traffic?
          </label>
          <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-700">
            {currentDescription}
          </span>
        </div>

        {/* 3 Quick-Select Segmented Options */}
        <div className="grid grid-cols-3 gap-1.5">
          <button
            type="button"
            onClick={() => onTrafficWeightChange(0.5)}
            className={`rounded-lg border px-2 py-1.5 text-center text-[11px] font-medium transition-all ${trafficWeight <= 1.0
                ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold shadow-xs'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
          >
            Shorter distance
          </button>
          <button
            type="button"
            onClick={() => onTrafficWeightChange(2.0)}
            className={`rounded-lg border px-2 py-1.5 text-center text-[11px] font-medium transition-all ${trafficWeight > 1.0 && trafficWeight <= 3.0
                ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold shadow-xs'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
          >
            Balanced
          </button>
          <button
            type="button"
            onClick={() => onTrafficWeightChange(4.0)}
            className={`rounded-lg border px-2 py-1.5 text-center text-[11px] font-medium transition-all ${trafficWeight > 3.0
                ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold shadow-xs'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
          >
            Avoid congestion
          </button>
        </div>

        {/* Fine-tuning Slider */}
        <div className="pt-1">
          <input
            id="traffic-weight-slider"
            type="range"
            min="0.0"
            max="5.0"
            step="0.5"
            value={trafficWeight}
            onChange={(e) => onTrafficWeightChange(parseFloat(e.target.value))}
            className="w-full accent-blue-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400 font-medium px-0.5 mt-0.5">
            <span>0.0 (Min)</span>
            <span>2.0 (Default)</span>
            <span>5.0 (Max)</span>
          </div>
        </div>

        <p className="text-[11px] text-slate-500 leading-relaxed">
          Higher settings may choose a longer route to reduce estimated congestion.
        </p>
      </div>

      {/* Mobility & Accessibility Mode 
      <div className="space-y-1.5 pt-2 border-t border-slate-100">
        <label className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
          <Footprints className="h-3.5 w-3.5 text-slate-400" />
          <span>Mobility & Accessibility Mode</span>
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            onClick={() => onAccessibilityModeChange?.('standard')}
            className={`rounded-lg border px-2 py-1.5 text-center text-[11px] font-medium transition-all cursor-pointer ${
              accessibilityMode === 'standard'
                ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold shadow-xs'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
            }`}
          >
            Standard Walking
          </button>
          <button
            type="button"
            onClick={() => onAccessibilityModeChange?.('step_free')}
            className={`rounded-lg border px-2 py-1.5 text-center text-[11px] font-medium transition-all cursor-pointer ${
              accessibilityMode === 'step_free'
                ? 'border-purple-500 bg-purple-50 text-purple-700 font-semibold shadow-xs'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
            }`}
          >
            Step-Free / Reduced
          </button>
        </div>
        <p className="text-[10px] text-slate-500 leading-tight">
          {accessibilityMode === 'step_free'
            ? 'Avoids stairs and verified physical obstacles. Accessibility coverage is incomplete: missing obstacle information does not mean a route is accessible.'
            : 'Standard pedestrian route. Stairs and ordinary footpaths are permitted.'}
        </p>
      </div>*/}
    </div>
  )
}
