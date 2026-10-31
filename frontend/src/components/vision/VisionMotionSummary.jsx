import { useState } from 'react'
import {
  Gauge,
  Move,
  Layers,
  Info,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Timer,
} from 'lucide-react'

export default function VisionMotionSummary({ motionSummary }) {
  const [showTechDetails, setShowTechDetails] = useState(false)

  if (!motionSummary) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
        <div className="flex items-center gap-2.5 text-xs font-semibold text-slate-500">
          <Info className="h-4 w-4 text-slate-400 shrink-0" />
          <span>Vehicle Motion Analysis: Not available for this analysis run.</span>
        </div>
      </div>
    )
  }

  const {
    total_tracks_analyzed: totalTracks = 0,
    valid_moving_tracks_count: validMoving = 0,
    stationary_tracks_count: stationary = 0,
    insufficient_data_tracks_count: insufficient = 0,
    average_scene_pixel_speed_px_per_sec: avgSceneSpeed,
    weighted_scene_pixel_speed_px_per_sec: weightedSceneSpeed,
    is_physical_speed_available: isPhysicalSpeedAvailable = false,
  } = motionSummary

  const movingPct =
    totalTracks > 0 ? Math.round((validMoving / totalTracks) * 100) : 0
  const stationaryPct =
    totalTracks > 0 ? Math.round((stationary / totalTracks) * 100) : 0
  const insufficientPct =
    totalTracks > 0
      ? Math.max(0, 100 - movingPct - stationaryPct)
      : 0

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 sm:p-5 shadow-xs space-y-4">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-blue-600 text-white shadow-2xs">
              <Gauge className="h-3.5 w-3.5" />
            </span>
            <h3 className="text-sm font-bold text-slate-900">
              Vehicle Motion Analysis
            </h3>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-700">
              {totalTracks} Tracks Analyzed
            </span>
          </div>
          <p className="text-xs text-slate-500 max-w-2xl leading-relaxed">
            Movement is measured from changes in vehicle positions across video frames in 2D image pixels per second (px/s). Physical speed in km/h is unavailable because the camera has not been calibrated.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 self-start sm:self-auto shrink-0">
          <span className="rounded-md bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-600 border border-slate-200/60">
            Unit: px/s (Image Space)
          </span>
          <span className="rounded-md bg-amber-50 border border-amber-200 px-2.5 py-1 text-[11px] font-semibold text-amber-800 flex items-center gap-1">
            <ShieldAlert className="h-3 w-3 text-amber-600" />
            <span>Physical Speed: {isPhysicalSpeedAvailable ? 'Available' : 'Uncalibrated'}</span>
          </span>
        </div>
      </div>

      {/* Main Motion Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* 1. Moving Tracks */}
        <div className="rounded-xl border border-emerald-200 bg-gradient-to-br from-emerald-50/70 to-white p-3.5 flex flex-col justify-between shadow-2xs">
          <div>
            <div className="flex items-center justify-between gap-1.5 mb-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-white shadow-2xs">
                <Move className="h-3.5 w-3.5" />
              </div>
              <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-emerald-100 text-emerald-800">
                {movingPct}% of tracks
              </span>
            </div>
            <div className="text-2xl font-black text-slate-900 font-mono">
              {validMoving.toLocaleString()}
            </div>
            <div className="text-xs font-bold text-slate-900 mt-1">
              Moving Tracks
            </div>
            <p className="text-xs text-slate-600 mt-1 leading-snug">
              Tracks showing noticeable movement in the video.
            </p>
          </div>
        </div>

        {/* 2. Stationary-Classified Tracks */}
        <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-slate-50/80 to-white p-3.5 flex flex-col justify-between shadow-2xs">
          <div>
            <div className="flex items-center justify-between gap-1.5 mb-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-600 text-white shadow-2xs">
                <Layers className="h-3.5 w-3.5" />
              </div>
              <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-slate-200 text-slate-700">
                {stationaryPct}% of tracks
              </span>
            </div>
            <div className="text-2xl font-black text-slate-900 font-mono">
              {stationary.toLocaleString()}
            </div>
            <div className="text-xs font-bold text-slate-900 mt-1">
              Stationary-Classified Tracks
            </div>
            <p className="text-xs text-slate-600 mt-1 leading-snug">
              Estimated from very small image movement (≤ 3.0 px), not proof that a vehicle stopped.
            </p>
          </div>
        </div>

        {/* 3. Insufficient Observations */}
        <div className="rounded-xl border border-amber-200 bg-gradient-to-br from-amber-50/70 to-white p-3.5 flex flex-col justify-between shadow-2xs">
          <div>
            <div className="flex items-center justify-between gap-1.5 mb-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-500 text-white shadow-2xs">
                <Layers className="h-3.5 w-3.5" />
              </div>
              <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-amber-100 text-amber-800">
                {insufficientPct}% of tracks
              </span>
            </div>
            <div className="text-2xl font-black text-slate-900 font-mono">
              {insufficient.toLocaleString()}
            </div>
            <div className="text-xs font-bold text-slate-900 mt-1">
              Insufficient Observations
            </div>
            <p className="text-xs text-slate-600 mt-1 leading-snug">
              Tracks with fewer than 2 frames, not enough to calculate movement.
            </p>
          </div>
        </div>

        {/* 4. Movement Rate */}
        <div className="rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50/70 to-white p-3.5 flex flex-col justify-between shadow-2xs">
          <div>
            <div className="flex items-center justify-between gap-1.5 mb-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600 text-white shadow-2xs">
                <Gauge className="h-3.5 w-3.5" />
              </div>
              <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-blue-100 text-blue-800">
                Image Movement
              </span>
            </div>
            <div className="text-2xl font-black text-blue-950 font-mono">
              {typeof avgSceneSpeed === 'number'
                ? `${avgSceneSpeed.toFixed(1)} px/s`
                : 'Not available'}
            </div>
            <div className="text-xs font-bold text-slate-900 mt-1">
              Movement Rate
            </div>
            <p className="text-xs text-slate-600 mt-1 leading-snug">
              Average image movement rate across moving tracks.
            </p>
          </div>

          {typeof weightedSceneSpeed === 'number' && (
            <div className="mt-2.5 pt-2 border-t border-blue-100/80 flex items-center justify-between text-xs">
              <span className="text-[10px] text-slate-500 font-medium">Duration-Weighted:</span>
              <span className="font-mono font-bold text-blue-900 text-xs">
                {weightedSceneSpeed.toFixed(1)} px/s
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Visual Composition Proportion Bar */}
      {totalTracks > 0 && (
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between text-[11px] text-slate-500 font-medium">
            <span>Movement Distribution</span>
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Moving: {validMoving} ({movingPct}%)
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-slate-400" />
                Stationary: {stationary} ({stationaryPct}%)
              </span>
              {insufficient > 0 && (
                <span className="flex items-center gap-1">
                  <span className="h-2 w-2 rounded-full bg-amber-400" />
                  Single Observation: {insufficient} ({insufficientPct}%)
                </span>
              )}
            </div>
          </div>
          <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden flex shadow-2xs">
            <div
              style={{ width: `${movingPct}%` }}
              className="bg-emerald-500 transition-all duration-300"
              title={`Moving: ${validMoving} tracks (${movingPct}%)`}
            />
            <div
              style={{ width: `${stationaryPct}%` }}
              className="bg-slate-400 transition-all duration-300"
              title={`Stationary: ${stationary} tracks (${stationaryPct}%)`}
            />
            <div
              style={{ width: `${insufficientPct}%` }}
              className="bg-amber-400 transition-all duration-300"
              title={`Single Observation: ${insufficient} tracks (${insufficientPct}%)`}
            />
          </div>
        </div>
      )}

      {/* Optional Technical Details Accordion */}
      <div className="rounded-xl border border-slate-200 overflow-hidden bg-slate-50/60">
        <button
          type="button"
          onClick={() => setShowTechDetails(!showTechDetails)}
          aria-expanded={showTechDetails}
          className="w-full flex items-center justify-between p-3 sm:px-4 text-xs font-semibold text-slate-700 hover:text-slate-900 hover:bg-slate-100/60 transition-colors cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-slate-400" />
            <span>View Motion Details &amp; Calculations</span>
          </div>
          <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
            <span>{showTechDetails ? 'Hide details' : 'Show details'}</span>
            {showTechDetails ? (
              <ChevronUp className="h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </div>
        </button>

        {showTechDetails && (
          <div className="p-4 sm:p-5 pt-2 border-t border-slate-200 bg-white space-y-4 text-xs text-slate-700">
            {/* Formulas & Aggregation */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/70 space-y-1">
                <div className="flex items-center gap-1.5 font-bold text-slate-900">
                  <Gauge className="h-3.5 w-3.5 text-blue-600" />
                  <span>Average Movement Rate Formula</span>
                </div>
                <p className="font-mono text-[11px] text-blue-900 bg-blue-50/60 p-1.5 rounded border border-blue-100">
                  Σ(v_track) / N_moving
                </p>
                <p className="text-[11px] text-slate-600 leading-relaxed">
                  Arithmetic mean of per-track rates across valid moving tracks. Weights each tracked vehicle equally regardless of duration.
                </p>
              </div>

              <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/70 space-y-1">
                <div className="flex items-center gap-1.5 font-bold text-slate-900">
                  <Timer className="h-3.5 w-3.5 text-indigo-600" />
                  <span>Duration-Weighted Rate Formula</span>
                </div>
                <p className="font-mono text-[11px] text-indigo-900 bg-indigo-50/60 p-1.5 rounded border border-indigo-100">
                  Σ(cumulative_displacement_px) / Σ(observed_duration_s)
                </p>
                <p className="text-[11px] text-slate-600 leading-relaxed">
                  Total pixel displacement divided by total observed duration across all moving tracks. Proportional to time spent in frame.
                </p>
              </div>
            </div>

            {/* Thresholds & Heuristics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px]">
              <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/70 space-y-1">
                <strong className="block font-semibold text-slate-900">
                  Stationary Threshold (≤ 3.0 px)
                </strong>
                <p className="text-slate-600 leading-relaxed">
                  Tracks with centroid displacement ≤ 3.0 px are classified as stationary to filter detection jitter. This is an image-space heuristic and does not prove ground-truth physical stopping (e.g., vehicles moving directly along the camera optical axis).
                </p>
              </div>

              <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/70 space-y-1">
                <strong className="block font-semibold text-slate-900">
                  Minimum Observation Threshold (2 Frames)
                </strong>
                <p className="text-slate-600 leading-relaxed">
                  Velocity calculations require at least 2 sequential frame observations. Tracks observed in only a single frame ({insufficient} tracks) are classified as single observations with no movement rate calculated.
                </p>
              </div>
            </div>

            {/* Calibration & Scientific Limits */}
            <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/60 space-y-1 text-[11px] text-amber-900">
              <div className="flex items-center gap-1.5 font-bold text-amber-950">
                <ShieldAlert className="h-3.5 w-3.5 text-amber-600" />
                <span>Camera Calibration &amp; Physical Speed Scope</span>
              </div>
              <p className="leading-relaxed text-amber-900/90">
                Converting image-plane pixel speeds (px/s) into physical velocity (km/h) requires camera calibration (homography matrix <em>H</em>), lens distortion correction, and ground control points. Uncalibrated video measurements cannot establish physical speed or road congestion.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

