import {
  Gauge,
  Move,
  Timer,
  AlertCircle,
  Layers,
  Info,
  ShieldAlert,
} from 'lucide-react'

export default function VisionMotionSummary({ motionSummary }) {
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
      {/* Section Header with Elevated Context */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-blue-600 text-white shadow-2xs">
              <Gauge className="h-3.5 w-3.5" />
            </span>
            <h3 className="text-sm font-bold text-slate-900">
              Vehicle Motion Analysis & Scene Movement
            </h3>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-700">
              {totalTracks} Tracks Evaluated
            </span>
          </div>
          <p className="text-xs text-slate-500 max-w-3xl leading-relaxed">
            2D image-plane displacement, duration, and movement rates calculated across consecutive YOLO + ByteTrack detections. Measurements exist strictly in sensor coordinates.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 self-start sm:self-auto shrink-0">
          <span className="rounded-md bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-600 border border-slate-200/60">
            Units: Displacement (px) • Rate (px/s)
          </span>
          <span className="rounded-md bg-amber-50 border border-amber-200 px-2.5 py-1 text-[11px] font-semibold text-amber-800 flex items-center gap-1">
            <ShieldAlert className="h-3 w-3 text-amber-600" />
            <span>Physical Speed: {isPhysicalSpeedAvailable ? 'Available' : 'Uncalibrated'}</span>
          </span>
        </div>
      </div>

      {/* Tier 1: Primary Scene Movement Rates (Featured Dynamic Metrics) */}
      <div>
        <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
          Primary Scene Movement Rates
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          {/* Average Image-Plane Movement Rate */}
          <div className="rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50/80 via-blue-50/40 to-white p-4 flex flex-col justify-between shadow-2xs relative overflow-hidden">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600 text-white shadow-2xs">
                    <Gauge className="h-4 w-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-blue-950">
                      Average Image-Plane Movement Rate
                    </h4>
                    <span className="text-[10px] font-medium text-blue-700">
                      Unweighted Track Mean
                    </span>
                  </div>
                </div>
                <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-bold text-blue-700">
                  Track Mean (px/s)
                </span>
              </div>

              <div className="pt-1">
                <div className="text-2xl sm:text-3xl font-black tracking-tight text-blue-950 font-mono">
                  {typeof avgSceneSpeed === 'number'
                    ? `${avgSceneSpeed.toFixed(1)} px/s`
                    : 'Not available'}
                </div>
                <p className="text-[11px] text-slate-600 mt-1 leading-snug">
                  Arithmetic mean of average pixel speeds across valid moving tracks. Weights each vehicle trajectory equally regardless of duration.
                </p>
              </div>
            </div>
            <div className="mt-2.5 pt-2 border-t border-blue-100 flex items-center justify-between text-[10px] text-blue-800/80 font-mono">
              <span>Formula: Σ(v_track) / N_moving</span>
              <span>Moving tracks: {validMoving}</span>
            </div>
          </div>

          {/* Duration-Weighted Scene Movement Rate */}
          <div className="rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-50/80 via-indigo-50/40 to-white p-4 flex flex-col justify-between shadow-2xs relative overflow-hidden">
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-white shadow-2xs">
                    <Timer className="h-4 w-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-indigo-950">
                      Duration-Weighted Scene Movement Rate
                    </h4>
                    <span className="text-[10px] font-medium text-indigo-700">
                      Time-Weighted Aggregate
                    </span>
                  </div>
                </div>
                <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-bold text-indigo-700">
                  Duration-Weighted (px/s)
                </span>
              </div>

              <div className="pt-1">
                <div className="text-2xl sm:text-3xl font-black tracking-tight text-indigo-950 font-mono">
                  {typeof weightedSceneSpeed === 'number'
                    ? `${weightedSceneSpeed.toFixed(1)} px/s`
                    : 'Not available'}
                </div>
                <p className="text-[11px] text-slate-600 mt-1 leading-snug">
                  Total cumulative pixel displacement divided by total observed duration across moving tracks. Proportional to time spent in frame.
                </p>
              </div>
            </div>
            <div className="mt-2.5 pt-2 border-t border-indigo-100 flex items-center justify-between text-[10px] text-indigo-800/80 font-mono">
              <span>Formula: Σ(disp_px) / Σ(dur_s)</span>
              <span>Mitigates transient noise</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tier 2: Track Classification & Composition */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Trajectory Classification Distribution
          </div>
          {totalTracks > 0 && (
            <div className="flex items-center gap-3 text-[10px] font-semibold text-slate-500">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Moving: {movingPct}%
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-slate-400" />
                Stationary: {stationaryPct}%
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-amber-400" />
                Single Obs: {insufficientPct}%
              </span>
            </div>
          )}
        </div>

        {/* Visual Composition Proportion Bar */}
        {totalTracks > 0 && (
          <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden flex mb-3 shadow-2xs">
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
              title={`Insufficient Data: ${insufficient} tracks (${insufficientPct}%)`}
            />
          </div>
        )}

        {/* 3 Classification Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* 1. Valid Moving Tracks */}
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-3.5 flex flex-col justify-between shadow-2xs">
            <div>
              <div className="flex items-center justify-between gap-1.5 mb-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-white shadow-2xs">
                  <Move className="h-3.5 w-3.5" />
                </div>
                <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-emerald-100 text-emerald-800">
                  Active Movement
                </span>
              </div>
              <div className="text-2xl font-black text-slate-900 font-mono">
                {validMoving.toLocaleString()}
              </div>
              <div className="text-xs font-bold text-slate-800 mt-0.5">
                Valid Moving Tracks
              </div>
              <div className="text-[11px] text-slate-600 mt-1 leading-snug">
                Actively displaced &gt; 3.0 px beyond jitter threshold ({movingPct}% of scene).
              </div>
            </div>
          </div>

          {/* 2. Stationary Tracks */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3.5 flex flex-col justify-between shadow-2xs">
            <div>
              <div className="flex items-center justify-between gap-1.5 mb-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-600 text-white shadow-2xs">
                  <Layers className="h-3.5 w-3.5" />
                </div>
                <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-slate-200 text-slate-700">
                  Image Jitter Heuristic
                </span>
              </div>
              <div className="text-2xl font-black text-slate-900 font-mono">
                {stationary.toLocaleString()}
              </div>
              <div className="text-xs font-bold text-slate-800 mt-0.5">
                Stationary-Classified Tracks
              </div>
              <div className="text-[11px] text-slate-600 mt-1 leading-snug">
                Centroid displacement within ≤ 3.0 px image jitter threshold.
              </div>
            </div>
            <div className="mt-2 text-[10px] text-slate-500 italic bg-white/70 p-1.5 rounded border border-slate-200/60 leading-tight">
              Heuristic only — does not establish ground-truth physical stopping.
            </div>
          </div>

          {/* 3. Insufficient Data Tracks */}
          <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-3.5 flex flex-col justify-between shadow-2xs">
            <div>
              <div className="flex items-center justify-between gap-1.5 mb-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-600 text-white shadow-2xs">
                  <AlertCircle className="h-3.5 w-3.5" />
                </div>
                <span className="rounded-full px-2 py-0.5 text-[10px] font-bold bg-amber-100 text-amber-800">
                  Single Observation
                </span>
              </div>
              <div className="text-2xl font-black text-slate-900 font-mono">
                {insufficient.toLocaleString()}
              </div>
              <div className="text-xs font-bold text-slate-800 mt-0.5">
                Insufficient Observations
              </div>
              <div className="text-[11px] text-slate-600 mt-1 leading-snug">
                &lt; 2 usable frame detections or zero elapsed observation duration.
              </div>
            </div>
            <div className="mt-2 text-[10px] text-amber-800/80 italic bg-white/70 p-1.5 rounded border border-amber-200/60 leading-tight">
              Minimum 2 sequential frames required for velocity calculations.
            </div>
          </div>
        </div>
      </div>

      {/* Educational Notice Banner & Calibration Disclaimer */}
      <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-3.5 sm:p-4 text-xs text-blue-950 space-y-2">
        <div className="flex items-center gap-2 font-bold text-blue-950">
          <Info className="h-4 w-4 text-blue-700 shrink-0" />
          <span>Image-Space Measurement Scope & Calibration Requirements</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-[11px] text-blue-900/90 leading-relaxed">
          <div className="bg-white/60 p-2.5 rounded-lg border border-blue-100 space-y-1">
            <strong className="text-blue-950 block font-semibold">
              1. Sensor Coordinate Scope (px &amp; px/s):
            </strong>
            Pixel displacement measures 2D movement across the image sensor. Perspective foreshortening causes distant vehicles to move fewer pixels per physical meter than foreground vehicles at the identical physical speed.
          </div>
          <div className="bg-white/60 p-2.5 rounded-lg border border-blue-100 space-y-1">
            <strong className="text-blue-950 block font-semibold">
              2. No Physical Speed or Road Congestion:
            </strong>
            Physical velocity (km/h or m/s) and traffic congestion cannot be established without camera-to-road homography (<em>H</em>), lens correction, and surveyed ground control points. These measurements do not alter AccessRoute route costs.
          </div>
        </div>
      </div>
    </div>
  )
}
