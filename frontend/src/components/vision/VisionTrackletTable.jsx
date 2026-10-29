import { useState, Fragment } from 'react'
import {
  Search,
  ChevronDown,
  ChevronUp,
  Check,
  AlertCircle,
  Move,
  Layers,
  Gauge,
  Info,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'

function formatPxSpeed(val, isStationary = false) {
  if (typeof val === 'number' && Number.isFinite(val)) {
    return `${val.toFixed(1)} px/s`
  }
  if (isStationary) {
    return '0.0 px/s'
  }
  return 'Not available'
}

function formatDisplacement(val) {
  if (typeof val === 'number' && Number.isFinite(val)) {
    return `${val.toFixed(1)} px`
  }
  return 'Not available'
}

function formatDuration(val) {
  if (typeof val === 'number' && Number.isFinite(val)) {
    return `${val.toFixed(2)}s`
  }
  return 'Not available'
}

function formatCount(val) {
  if (typeof val === 'number' && Number.isFinite(val)) {
    return val.toLocaleString()
  }
  return 'Not available'
}

export default function VisionTrackletTable({
  tracklets,
  reclassifiedDetections,
  minLifespan,
}) {
  const [classFilter, setClassFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all') // 'all' | 'retained' | 'filtered'
  const [motionFilter, setMotionFilter] = useState('all') // 'all' | 'moving' | 'stationary' | 'insufficient'
  const [searchTerm, setSearchTerm] = useState('')
  const [isExpanded, setIsExpanded] = useState(true)
  const [expandedTrackId, setExpandedTrackId] = useState(null)

  if (!tracklets || tracklets.length === 0) return null

  const retainedCount = tracklets.filter((t) => !t.is_filtered).length
  const filteredCount = tracklets.filter((t) => t.is_filtered).length

  const movingCount = tracklets.filter(
    (t) => t.motion?.measurement_status === 'valid',
  ).length
  const stationaryCount = tracklets.filter(
    (t) => t.motion?.measurement_status === 'stationary',
  ).length
  const insufficientCount = tracklets.filter(
    (t) => t.motion?.measurement_status === 'insufficient_observations',
  ).length

  const filteredTracklets = tracklets.filter((t) => {
    if (classFilter !== 'all' && t.assigned_class.toLowerCase() !== classFilter) {
      return false
    }
    if (statusFilter === 'retained' && t.is_filtered) {
      return false
    }
    if (statusFilter === 'filtered' && !t.is_filtered) {
      return false
    }
    if (motionFilter === 'moving' && t.motion?.measurement_status !== 'valid') {
      return false
    }
    if (
      motionFilter === 'stationary' &&
      t.motion?.measurement_status !== 'stationary'
    ) {
      return false
    }
    if (
      motionFilter === 'insufficient' &&
      t.motion?.measurement_status !== 'insufficient_observations'
    ) {
      return false
    }
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      const matchesId = String(t.track_id).includes(term)
      const matchesClass = t.assigned_class.toLowerCase().includes(term)
      const matchesStatus = t.motion?.measurement_status
        ?.toLowerCase()
        .includes(term)
      if (!matchesId && !matchesClass && !matchesStatus) return false
    }
    return true
  })

  return (
    <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-xs space-y-0">
      {/* Table Header Toggle */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between p-4 sm:px-5 cursor-pointer bg-slate-50/70 hover:bg-slate-50 transition-colors border-b border-slate-100"
      >
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-bold text-slate-900">
            Tracklet Trajectory & Motion Details ({retainedCount} Retained,{' '}
            {filteredCount} Filtered / {tracklets.length} Total Tracks)
          </h3>
          <span className="rounded-full bg-slate-200/80 px-2 py-0.5 text-[10px] font-bold text-slate-700">
            ByteTrack + Motion Analysis
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 hidden sm:inline">
            {isExpanded ? 'Collapse table' : 'Expand table'}
          </span>
          <button
            type="button"
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600"
            aria-label={isExpanded ? 'Collapse table' : 'Expand table'}
          >
            {isExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 sm:p-5 space-y-4">
          {/* Discovery Banner for Per-Track Motion Details */}
          <div className="flex items-center justify-between gap-2 rounded-xl border border-blue-100 bg-blue-50/50 px-3.5 py-2 text-xs text-blue-900 shadow-2xs">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-blue-600 shrink-0" />
              <span>
                <strong>Inspect Track Details:</strong> Click any vehicle row or
                the <strong>Inspect</strong> button to open granular
                displacement (px), image-plane movement rates (px/s), and
                consecutive step transitions.
              </span>
            </div>
            <span className="hidden sm:inline-block text-[11px] font-semibold text-blue-700 shrink-0">
              {filteredTracklets.length} tracks shown
            </span>
          </div>

          {/* Controls: Search & Category / Motion Filters */}
          <div className="flex flex-wrap items-center justify-between gap-2.5">
            <div className="flex flex-wrap items-center gap-2">
              {/* Category Pills */}
              <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-xs">
                {['all', 'car', 'bus', 'motorcycle', 'truck'].map((cls) => (
                  <button
                    key={cls}
                    type="button"
                    onClick={() => setClassFilter(cls)}
                    className={`capitalize px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                      classFilter === cls
                        ? 'bg-white text-blue-700 shadow-2xs font-bold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {cls}
                  </button>
                ))}
              </div>

              {/* Status Filter */}
              <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-xs">
                {[
                  { id: 'all', label: `All (${tracklets.length})` },
                  { id: 'retained', label: `Retained (${retainedCount})` },
                  { id: 'filtered', label: `Filtered (${filteredCount})` },
                ].map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setStatusFilter(s.id)}
                    className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                      statusFilter === s.id
                        ? 'bg-white text-slate-900 shadow-2xs font-bold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              {/* Motion Filter Pills */}
              <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-xs">
                {[
                  { id: 'all', label: 'All Motion' },
                  { id: 'moving', label: `Moving (${movingCount})` },
                  { id: 'stationary', label: `Stationary (${stationaryCount})` },
                  {
                    id: 'insufficient',
                    label: `Single Obs (${insufficientCount})`,
                  },
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setMotionFilter(m.id)}
                    className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                      motionFilter === m.id
                        ? 'bg-white text-indigo-700 shadow-2xs font-bold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Search Input */}
            <div className="relative min-w-[180px]">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search track ID, class, status…"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-white pl-8 pr-3 py-1.5 text-xs text-slate-800 placeholder-slate-400 focus:border-blue-500 focus:outline-none shadow-2xs"
              />
            </div>
          </div>

          {/* Table Container */}
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold text-[10px] uppercase tracking-wider">
                  <th className="py-2.5 px-3">Track ID</th>
                  <th className="py-2.5 px-3">Assigned Class</th>
                  <th className="py-2.5 px-3">Motion Status</th>
                  <th className="py-2.5 px-3 text-right">
                    Movement Rate (px/s)
                  </th>
                  <th className="py-2.5 px-3">Raw Predictions</th>
                  <th className="py-2.5 px-3 text-right">Detections</th>
                  <th className="py-2.5 px-3 text-right">Observed Frames</th>
                  <th className="py-2.5 px-3 text-right">Avg Confidence</th>
                  <th className="py-2.5 px-3 text-center">Post-Proc Status</th>
                  <th className="py-2.5 px-3 text-center">Motion Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredTracklets.length === 0 ? (
                  <tr>
                    <td colSpan={10} className="py-8 text-center text-slate-400">
                      No tracklets match the selected filters.
                    </td>
                  </tr>
                ) : (
                  filteredTracklets.map((t) => {
                    const isExpandedRow = expandedTrackId === t.track_id
                    const motion = t.motion
                    const isStationary = motion?.is_stationary ?? false

                    return (
                      <Fragment key={t.track_id}>
                        <tr
                          onClick={() =>
                            setExpandedTrackId(
                              isExpandedRow ? null : t.track_id,
                            )
                          }
                          className={`hover:bg-blue-50/40 transition-colors cursor-pointer ${
                            t.is_filtered ? 'opacity-60 bg-slate-50/40' : ''
                          } ${isExpandedRow ? 'bg-blue-50/30' : ''}`}
                        >
                          {/* Track ID */}
                          <td className="py-2.5 px-3 font-mono font-bold text-slate-900">
                            #{t.track_id}
                          </td>

                          {/* Assigned Class */}
                          <td className="py-2.5 px-3">
                            <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-bold capitalize bg-blue-50 text-blue-700 border border-blue-200/60">
                              {t.assigned_class}
                            </span>
                          </td>

                          {/* Motion Status */}
                          <td className="py-2.5 px-3">
                            {!motion ? (
                              <span className="rounded-full bg-slate-100 text-slate-600 px-2 py-0.5 text-[10px] font-semibold">
                                Not available
                              </span>
                            ) : motion.measurement_status === 'valid' ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 text-[10px] font-bold">
                                <Move className="h-2.5 w-2.5" />
                                <span>Moving</span>
                              </span>
                            ) : motion.measurement_status === 'stationary' ? (
                              <span
                                className="inline-flex items-center gap-1 rounded-full bg-slate-200/90 text-slate-700 px-2 py-0.5 text-[10px] font-bold"
                                title="Image jitter heuristic (centroid displacement <= 3.0 px)"
                              >
                                <Layers className="h-2.5 w-2.5" />
                                <span>Stationary</span>
                              </span>
                            ) : motion.measurement_status ===
                              'insufficient_observations' ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 text-amber-800 px-2 py-0.5 text-[10px] font-bold">
                                <AlertCircle className="h-2.5 w-2.5" />
                                <span>Single Obs</span>
                              </span>
                            ) : (
                              <span className="rounded-full bg-slate-100 text-slate-700 px-2 py-0.5 text-[10px] font-semibold">
                                {motion.measurement_status}
                              </span>
                            )}
                          </td>

                          {/* Image-Plane Movement Rate (px/s) */}
                          <td className="py-2.5 px-3 text-right font-mono font-semibold text-slate-800">
                            {formatPxSpeed(
                              motion?.average_pixel_speed_px_per_sec,
                              isStationary,
                            )}
                          </td>

                          {/* Raw Predictions */}
                          <td className="py-2.5 px-3 text-slate-600 font-mono text-[11px]">
                            {Object.entries(t.original_classes || {})
                              .map(([cls, count]) => `${cls}: ${count}`)
                              .join(', ')}
                            {t.has_class_switch && (
                              <span
                                className="ml-1.5 inline-block text-[10px] text-amber-700 font-semibold bg-amber-50 px-1.5 py-0.2 rounded border border-amber-200"
                                title="Multiple raw classes were observed. Resolved to majority vote."
                              >
                                Oscillated
                              </span>
                            )}
                          </td>

                          {/* Detections */}
                          <td className="py-2.5 px-3 text-right font-semibold text-slate-800">
                            {t.detection_count}
                          </td>

                          {/* Observed Frames & Lifespan */}
                          <td className="py-2.5 px-3 text-right text-slate-500 font-mono text-[11px]">
                            f.{t.first_frame}–f.{t.last_frame} (
                            {t.lifespan_frames} frames)
                          </td>

                          {/* Avg Confidence */}
                          <td className="py-2.5 px-3 text-right font-mono font-semibold text-slate-700">
                            {(t.average_confidence * 100).toFixed(1)}%
                          </td>

                          {/* Post-Proc Status */}
                          <td className="py-2.5 px-3 text-center">
                            {t.is_filtered ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 text-amber-800 px-2 py-0.5 text-[10px] font-bold">
                                <AlertCircle className="h-3 w-3" />
                                <span>Filtered (&lt; {minLifespan} frames)</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 text-[10px] font-bold">
                                <Check className="h-3 w-3" />
                                <span>Retained</span>
                              </span>
                            )}
                          </td>

                          {/* Details Toggle Button */}
                          <td className="py-2.5 px-3 text-center">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation()
                                setExpandedTrackId(
                                  isExpandedRow ? null : t.track_id,
                                )
                              }}
                              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all ${
                                isExpandedRow
                                  ? 'bg-blue-600 text-white shadow-2xs'
                                  : 'bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200/80 shadow-2xs'
                              }`}
                              aria-expanded={isExpandedRow}
                            >
                              <span>{isExpandedRow ? 'Close' : 'Inspect'}</span>
                              {isExpandedRow ? (
                                <ChevronUp className="h-3 w-3" />
                              ) : (
                                <ChevronDown className="h-3 w-3" />
                              )}
                            </button>
                          </td>
                        </tr>

                        {/* Expandable Motion Details Row */}
                        {isExpandedRow && (
                          <tr className="bg-slate-50/90 border-b border-slate-200">
                            <td colSpan={10} className="p-4 sm:p-5 space-y-4">
                              {/* Expanded Row Header */}
                              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200/80 pb-3">
                                <div className="space-y-0.5">
                                  <div className="flex items-center gap-2">
                                    <span className="flex h-5 w-5 items-center justify-center rounded-md bg-blue-600 text-white shadow-2xs">
                                      <Gauge className="h-3 w-3" />
                                    </span>
                                    <h4 className="text-xs font-bold text-slate-900">
                                      Track #{t.track_id} Trajectory &amp; Motion
                                      Analysis Details
                                    </h4>
                                    <span className="rounded-md bg-blue-100 text-blue-800 px-2 py-0.5 text-[10px] font-bold capitalize">
                                      {t.assigned_class}
                                    </span>
                                  </div>
                                  <p className="text-[11px] text-slate-500">
                                    Frame interval: f.
                                    {motion?.first_observed_frame ??
                                      t.first_frame}{' '}
                                    to f.
                                    {motion?.last_observed_frame ??
                                      t.last_frame}
                                    {motion?.first_observed_time_seconds != null &&
                                    motion?.last_observed_time_seconds != null
                                      ? ` (${motion.first_observed_time_seconds.toFixed(2)}s – ${motion.last_observed_time_seconds.toFixed(2)}s)`
                                      : ''}
                                  </p>
                                </div>

                                <div className="flex items-center gap-2">
                                  {motion?.status_message && (
                                    <span className="text-[11px] text-slate-600 italic bg-white px-2.5 py-1 rounded-md border border-slate-200 shadow-2xs">
                                      {motion.status_message}
                                    </span>
                                  )}
                                  <button
                                    type="button"
                                    onClick={() => setExpandedTrackId(null)}
                                    className="text-xs text-slate-400 hover:text-slate-600 px-2 py-1 rounded"
                                    aria-label="Close details"
                                  >
                                    ✕
                                  </button>
                                </div>
                              </div>

                              {!motion ? (
                                <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-500">
                                  Motion measurement is not available for this
                                  tracklet.
                                </div>
                              ) : (
                                <>
                                  {/* Dedicated Stationary Classification Heuristic Notice */}
                                  {motion.is_stationary && (
                                    <div className="rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-700 flex items-start gap-2 shadow-2xs">
                                      <Info className="h-4 w-4 text-slate-500 shrink-0 mt-0.5" />
                                      <div className="space-y-0.5 leading-relaxed">
                                        <strong className="text-slate-900 block font-semibold">
                                          Stationary Classification Heuristic:
                                        </strong>
                                        This vehicle was classified as stationary
                                        because its 2D centroid displacement was
                                        ≤ 3.0 px (within sensor jitter
                                        threshold). This is an image-space
                                        heuristic to filter detection jitter and
                                        does not prove ground-truth physical
                                        stopping (vehicles moving directly along
                                        the camera line-of-sight or in the far
                                        background may show low pixel motion).
                                      </div>
                                    </div>
                                  )}

                                  {/* 8-Card Motion Details Grid */}
                                  <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
                                    {/* 1. Track ID */}
                                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                        Track ID
                                      </div>
                                      <div className="text-sm font-black font-mono text-slate-900">
                                        #{t.track_id}
                                      </div>
                                      <div className="text-[10px] text-slate-500 capitalize truncate">
                                        {t.assigned_class}
                                      </div>
                                    </div>

                                    {/* 2. Motion Status */}
                                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                        Motion Status
                                      </div>
                                      <div className="text-xs font-bold text-slate-900 capitalize truncate">
                                        {motion.measurement_status.replace(
                                          '_',
                                          ' ',
                                        )}
                                      </div>
                                      <div className="text-[10px] text-slate-500 truncate">
                                        {motion.is_stationary
                                          ? '≤ 3 px jitter'
                                          : 'Displaced'}
                                      </div>
                                    </div>

                                    {/* 3. Usable Observations */}
                                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                        Usable Obs
                                      </div>
                                      <div className="text-sm font-black text-slate-900 font-mono">
                                        {formatCount(
                                          motion.usable_observation_count,
                                        )}
                                      </div>
                                      <div className="text-[10px] text-slate-500 truncate">
                                        of {t.detection_count} raw boxes
                                      </div>
                                    </div>

                                    {/* 4. Observed Duration (s) */}
                                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                        Duration (s)
                                      </div>
                                      <div className="text-sm font-black font-mono text-slate-900">
                                        {formatDuration(
                                          motion.total_observed_duration_seconds,
                                        )}
                                      </div>
                                      <div className="text-[10px] text-slate-500 truncate">
                                        Elapsed time
                                      </div>
                                    </div>

                                    {/* 5. Cumulative Displacement (px) */}
                                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                        Cumul. Disp (px)
                                      </div>
                                      <div className="text-sm font-black font-mono text-slate-900">
                                        {formatDisplacement(
                                          motion.cumulative_pixel_displacement,
                                        )}
                                      </div>
                                      <div className="text-[10px] text-slate-500 truncate">
                                        Total trajectory
                                      </div>
                                    </div>

                                    {/* 6. Net Displacement (px) */}
                                    <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                        Net Disp (px)
                                      </div>
                                      <div className="text-sm font-black font-mono text-slate-900">
                                        {formatDisplacement(
                                          motion.net_pixel_displacement,
                                        )}
                                      </div>
                                      <div className="text-[10px] text-slate-500 truncate">
                                        First-to-last
                                      </div>
                                    </div>

                                    {/* 7. Average Image-Plane Movement (px/s) */}
                                    <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-bold text-blue-800 uppercase">
                                        Avg Rate (px/s)
                                      </div>
                                      <div className="text-sm font-black font-mono text-blue-900">
                                        {formatPxSpeed(
                                          motion.average_pixel_speed_px_per_sec,
                                          isStationary,
                                        )}
                                      </div>
                                      <div className="text-[10px] text-blue-700/80 truncate">
                                        Movement rate
                                      </div>
                                    </div>

                                    {/* 8. Maximum Image-Plane Movement (px/s) */}
                                    <div className="rounded-lg border border-indigo-200 bg-indigo-50/50 p-2.5 space-y-0.5 shadow-2xs">
                                      <div className="text-[10px] font-bold text-indigo-800 uppercase">
                                        Peak Rate (px/s)
                                      </div>
                                      <div className="text-sm font-black font-mono text-indigo-900">
                                        {formatPxSpeed(
                                          motion.max_pixel_speed_px_per_sec,
                                          isStationary,
                                        )}
                                      </div>
                                      <div className="text-[10px] text-indigo-700/80 truncate">
                                        Max step (px/s)
                                      </div>
                                    </div>
                                  </div>

                                  {/* Physical Speed & Methodology Notice */}
                                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 rounded-lg border border-amber-200 bg-amber-50/60 p-3 text-[11px] text-amber-900">
                                    <div className="flex items-center gap-1.5 font-medium">
                                      <ShieldAlert className="h-4 w-4 text-amber-600 shrink-0" />
                                      <span>
                                        <strong>Physical Speed:</strong>{' '}
                                        {motion.is_physical_speed_available
                                          ? 'Available'
                                          : 'Not available — camera uncalibrated.'}
                                      </span>
                                    </div>
                                    <div className="text-amber-800/90 text-[10px]">
                                      Measurements are in 2D image pixels (px,
                                      px/s) and do not represent km/h or m/s.
                                    </div>
                                  </div>

                                  {/* Consecutive Step Displacements Breakdown (if available) */}
                                  {motion.step_displacements &&
                                    motion.step_displacements.length > 0 && (
                                      <div className="rounded-lg border border-slate-200 bg-white p-3 space-y-2">
                                        <div className="flex items-center justify-between text-xs font-bold text-slate-800">
                                          <span>
                                            Consecutive Step Displacements (
                                            {motion.step_displacements.length}{' '}
                                            Steps)
                                          </span>
                                          <span className="text-[10px] text-slate-500 font-normal">
                                            Euclidean Center Displacement
                                          </span>
                                        </div>
                                        <div className="overflow-x-auto max-h-48 rounded border border-slate-200">
                                          <table className="w-full text-left text-[11px] border-collapse">
                                            <thead>
                                              <tr className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                                                <th className="py-1 px-2.5">
                                                  Frame Interval
                                                </th>
                                                <th className="py-1 px-2.5">
                                                  Timestamp Span (s)
                                                </th>
                                                <th className="py-1 px-2.5 text-right">
                                                  Δ Time (s)
                                                </th>
                                                <th className="py-1 px-2.5 text-right">
                                                  Displacement (px)
                                                </th>
                                                <th className="py-1 px-2.5 text-right">
                                                  Step Rate (px/s)
                                                </th>
                                                <th className="py-1 px-2.5 text-center">
                                                  Timing Source
                                                </th>
                                              </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-100 font-mono">
                                              {motion.step_displacements.map(
                                                (step, sIdx) => (
                                                  <tr
                                                    key={sIdx}
                                                    className="hover:bg-slate-50/50"
                                                  >
                                                    <td className="py-1 px-2.5 text-slate-800">
                                                      f.{step.from_frame} → f.
                                                      {step.to_frame}
                                                    </td>
                                                    <td className="py-1 px-2.5 text-slate-600">
                                                      {step.from_timestamp_seconds.toFixed(
                                                        2,
                                                      )}
                                                      s →{' '}
                                                      {step.to_timestamp_seconds.toFixed(
                                                        2,
                                                      )}
                                                      s
                                                    </td>
                                                    <td className="py-1 px-2.5 text-right text-slate-700">
                                                      {step.elapsed_seconds.toFixed(
                                                        3,
                                                      )}
                                                      s
                                                    </td>
                                                    <td className="py-1 px-2.5 text-right font-semibold text-slate-900">
                                                      {step.displacement_pixels.toFixed(
                                                        1,
                                                      )}{' '}
                                                      px
                                                    </td>
                                                    <td className="py-1 px-2.5 text-right text-blue-700 font-semibold">
                                                      {step.pixel_speed_px_per_sec.toFixed(
                                                        1,
                                                      )}{' '}
                                                      px/s
                                                    </td>
                                                    <td className="py-1 px-2.5 text-center text-slate-500 text-[10px]">
                                                      {step.timing_source}
                                                    </td>
                                                  </tr>
                                                ),
                                              )}
                                            </tbody>
                                          </table>
                                        </div>
                                      </div>
                                    )}

                                  {/* Insufficient Observations Notice */}
                                  {motion.measurement_status ===
                                    'insufficient_observations' && (
                                    <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-2.5 text-xs text-amber-800">
                                      <Info className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                                      <span>
                                        This track was observed in fewer than 2
                                        usable frames. A minimum of 2 sequential
                                        observations is required to calculate
                                        displacement and image-plane speed.
                                      </span>
                                    </div>
                                  )}
                                </>
                              )}
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    )
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Reclassified Detections (Temporal Majority Voting Results) */}
          {reclassifiedDetections && reclassifiedDetections.length > 0 && (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-indigo-900">
                  Intra-Tracklet Label Corrections (
                  {reclassifiedDetections.length} Detections Smoothed)
                </span>
                <span className="text-[10px] text-indigo-600 font-medium">
                  Temporal Majority Voting
                </span>
              </div>
              <p className="text-[11px] text-indigo-800 leading-snug">
                The following individual detections had an unstable class
                prediction on an active track that was smoothed to the winning
                majority category:
              </p>
              <div className="overflow-x-auto rounded-lg border border-indigo-200 bg-white">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-indigo-100/60 text-indigo-900 font-bold text-[10px] uppercase">
                      <th className="py-1.5 px-2.5">Frame #</th>
                      <th className="py-1.5 px-2.5">Timestamp</th>
                      <th className="py-1.5 px-2.5">Track ID</th>
                      <th className="py-1.5 px-2.5">Raw Predicted Class</th>
                      <th className="py-1.5 px-2.5">Majority-Voted Class</th>
                      <th className="py-1.5 px-2.5 text-right">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-indigo-50">
                    {reclassifiedDetections.map((item, idx) => (
                      <tr key={idx} className="hover:bg-indigo-50/40">
                        <td className="py-1.5 px-2.5 font-mono text-slate-800">
                          f.{item.frame_number}
                        </td>
                        <td className="py-1.5 px-2.5 text-slate-600 font-mono">
                          {item.timestamp_seconds.toFixed(2)}s
                        </td>
                        <td className="py-1.5 px-2.5 font-mono font-bold text-slate-900">
                          #{item.track_id}
                        </td>
                        <td className="py-1.5 px-2.5">
                          <span className="line-through text-rose-600 font-medium mr-1.5">
                            {item.raw_class}
                          </span>
                        </td>
                        <td className="py-1.5 px-2.5">
                          <span className="font-bold text-emerald-700">
                            {item.assigned_class}
                          </span>
                        </td>
                        <td className="py-1.5 px-2.5 text-right font-mono text-slate-700">
                          {(item.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
