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
  const [expandedTechTrackId, setExpandedTechTrackId] = useState(null)
  const [showCorrections, setShowCorrections] = useState(false)

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
            Track Details ({retainedCount} Retained, {filteredCount} Filtered / {tracklets.length} Total)
          </h3>
          <span className="rounded-full bg-slate-200/80 px-2 py-0.5 text-[10px] font-bold text-slate-700">
            ByteTrack + Motion
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 hidden sm:inline">
            {isExpanded ? 'Collapse table' : 'Expand table'}
          </span>
          <button
            type="button"
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600 cursor-pointer"
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
                    className={`capitalize px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all cursor-pointer ${
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
                    className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all cursor-pointer ${
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
                  { id: 'all', label: 'All Movement' },
                  { id: 'moving', label: `Moving (${movingCount})` },
                  { id: 'stationary', label: `Stationary (${stationaryCount})` },
                  ...(insufficientCount > 0
                    ? [
                        {
                          id: 'insufficient',
                          label: `Single Obs (${insufficientCount})`,
                        },
                      ]
                    : []),
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setMotionFilter(m.id)}
                    className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all cursor-pointer ${
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
            <div className="relative min-w-[200px]">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search track ID, vehicle type…"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-white pl-8 pr-3 py-1.5 text-xs text-slate-800 placeholder-slate-400 focus:border-blue-500 focus:outline-none shadow-2xs"
              />
            </div>
          </div>

          {/* Table Container - Simplified 7 Columns */}
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold text-[10px] uppercase tracking-wider">
                  <th className="py-2.5 px-3">Track ID</th>
                  <th className="py-2.5 px-3">Vehicle Type</th>
                  <th className="py-2.5 px-3">Movement</th>
                  <th className="py-2.5 px-3 text-right">Movement Rate (px/s)</th>
                  <th className="py-2.5 px-3 text-right">Detections</th>
                  <th className="py-2.5 px-3 text-center">Status</th>
                  <th className="py-2.5 px-3 text-center">Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredTracklets.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-8 text-center text-slate-400">
                      No tracks match the selected filters.
                    </td>
                  </tr>
                ) : (
                  filteredTracklets.map((t) => {
                    const isExpandedRow = expandedTrackId === t.track_id
                    const isTechExpanded = expandedTechTrackId === t.track_id
                    const motion = t.motion
                    const isStationary = motion?.is_stationary ?? false

                    return (
                      <Fragment key={t.track_id}>
                        <tr
                          onClick={() => {
                            if (isExpandedRow) {
                              setExpandedTrackId(null)
                              setExpandedTechTrackId(null)
                            } else {
                              setExpandedTrackId(t.track_id)
                            }
                          }}
                          className={`hover:bg-blue-50/40 transition-colors cursor-pointer ${
                            t.is_filtered ? 'opacity-70 bg-slate-50/40' : ''
                          } ${isExpandedRow ? 'bg-blue-50/30' : ''}`}
                        >
                          {/* 1. Track ID */}
                          <td className="py-2.5 px-3 font-mono font-bold text-slate-900">
                            #{t.track_id}
                          </td>

                          {/* 2. Vehicle Type */}
                          <td className="py-2.5 px-3">
                            <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-bold capitalize bg-blue-50 text-blue-700 border border-blue-200/60">
                              {t.assigned_class}
                            </span>
                          </td>

                          {/* 3. Movement */}
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
                                title="Image movement <= 3.0 px"
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

                          {/* 4. Movement Rate (px/s) */}
                          <td className="py-2.5 px-3 text-right font-mono font-semibold text-slate-800">
                            {formatPxSpeed(
                              motion?.average_pixel_speed_px_per_sec,
                              isStationary,
                            )}
                          </td>

                          {/* 5. Detections */}
                          <td className="py-2.5 px-3 text-right font-semibold text-slate-800 font-mono">
                            {t.detection_count}
                          </td>

                          {/* 6. Status */}
                          <td className="py-2.5 px-3 text-center">
                            {t.is_filtered ? (
                              <span
                                className="inline-flex items-center gap-1 rounded-full bg-amber-100 text-amber-800 px-2 py-0.5 text-[10px] font-bold"
                                title={minLifespan ? `Filtered (< ${minLifespan} frames)` : 'Filtered'}
                              >
                                <AlertCircle className="h-3 w-3" />
                                <span>Filtered</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 text-[10px] font-bold">
                                <Check className="h-3 w-3" />
                                <span>Retained</span>
                              </span>
                            )}
                          </td>

                          {/* 7. Inspect Toggle */}
                          <td className="py-2.5 px-3 text-center">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation()
                                if (isExpandedRow) {
                                  setExpandedTrackId(null)
                                  setExpandedTechTrackId(null)
                                } else {
                                  setExpandedTrackId(t.track_id)
                                }
                              }}
                              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
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

                        {/* Compact Track Inspection Panel */}
                        {isExpandedRow && (
                          <tr className="bg-slate-50/90 border-b border-slate-200">
                            <td colSpan={7} className="p-4 sm:p-5 space-y-3.5">
                              {/* Panel Header */}
                              <div className="flex items-center justify-between gap-2 border-b border-slate-200/80 pb-2.5">
                                <div className="flex items-center gap-2">
                                  <span className="flex h-5 w-5 items-center justify-center rounded-md bg-blue-600 text-white shadow-2xs">
                                    <Gauge className="h-3 w-3" />
                                  </span>
                                  <h4 className="text-xs font-bold text-slate-900">
                                    Track #{t.track_id} Summary
                                  </h4>
                                  <span className="rounded-md bg-blue-100 text-blue-800 px-2 py-0.5 text-[10px] font-bold capitalize">
                                    {t.assigned_class}
                                  </span>
                                </div>

                                <button
                                  type="button"
                                  onClick={() => {
                                    setExpandedTrackId(null)
                                    setExpandedTechTrackId(null)
                                  }}
                                  className="text-xs text-slate-400 hover:text-slate-600 px-2 py-1 rounded cursor-pointer"
                                  aria-label="Close inspection"
                                >
                                  ✕
                                </button>
                              </div>

                              {/* 5 Main Summary Cards */}
                              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
                                {/* 1. Track ID & Vehicle Type */}
                                <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                  <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                    Track ID &amp; Type
                                  </div>
                                  <div className="text-sm font-black font-mono text-slate-900">
                                    #{t.track_id}
                                  </div>
                                  <div className="text-[11px] font-bold text-blue-700 capitalize truncate">
                                    {t.assigned_class}
                                  </div>
                                </div>

                                {/* 2. Movement Status */}
                                <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                  <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                    Movement
                                  </div>
                                  <div className="text-xs font-bold text-slate-900 capitalize truncate">
                                    {motion?.measurement_status === 'valid'
                                      ? 'Moving'
                                      : motion?.measurement_status === 'stationary'
                                      ? 'Stationary'
                                      : motion?.measurement_status ===
                                        'insufficient_observations'
                                      ? 'Single Obs'
                                      : 'Not available'}
                                  </div>
                                  <div className="text-[10px] text-slate-500 truncate">
                                    {motion?.is_stationary
                                      ? '≤ 3.0 px image jitter'
                                      : motion?.measurement_status === 'valid'
                                      ? 'Noticeable movement'
                                      : 'Fewer than 2 frames'}
                                  </div>
                                </div>

                                {/* 3. Movement Rate (px/s) */}
                                <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-2.5 space-y-0.5 shadow-2xs">
                                  <div className="text-[10px] font-bold text-blue-800 uppercase">
                                    Movement Rate
                                  </div>
                                  <div className="text-sm font-black font-mono text-blue-900">
                                    {formatPxSpeed(
                                      motion?.average_pixel_speed_px_per_sec,
                                      isStationary,
                                    )}
                                  </div>
                                  <div className="text-[10px] text-blue-700/80 truncate">
                                    Image pixels per second
                                  </div>
                                </div>

                                {/* 4. Number of Detections */}
                                <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                  <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                    Detections
                                  </div>
                                  <div className="text-sm font-black text-slate-900 font-mono">
                                    {t.detection_count}
                                  </div>
                                  <div className="text-[10px] text-slate-500 truncate">
                                    Frame detections
                                  </div>
                                </div>

                                {/* 5. Observation Duration */}
                                <div className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-0.5 shadow-2xs">
                                  <div className="text-[10px] font-semibold text-slate-500 uppercase">
                                    Duration
                                  </div>
                                  <div className="text-sm font-black font-mono text-slate-900">
                                    {formatDuration(
                                      motion?.total_observed_duration_seconds,
                                    )}
                                  </div>
                                  <div className="text-[10px] text-slate-500 truncate">
                                    f.{t.first_frame}–f.{t.last_frame}
                                  </div>
                                </div>
                              </div>

                              {/* Nested Technical Details Accordion */}
                              <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
                                <button
                                  type="button"
                                  onClick={() =>
                                    setExpandedTechTrackId(
                                      isTechExpanded ? null : t.track_id,
                                    )
                                  }
                                  aria-expanded={isTechExpanded}
                                  className="w-full flex items-center justify-between p-2.5 sm:px-3 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors cursor-pointer"
                                >
                                  <div className="flex items-center gap-1.5">
                                    <Info className="h-3.5 w-3.5 text-slate-400" />
                                    <span>View Technical Details</span>
                                  </div>
                                  <div className="flex items-center gap-1 text-[11px] text-slate-500">
                                    <span>
                                      {isTechExpanded
                                        ? 'Hide technical metrics'
                                        : 'Show technical metrics'}
                                    </span>
                                    {isTechExpanded ? (
                                      <ChevronUp className="h-3.5 w-3.5" />
                                    ) : (
                                      <ChevronDown className="h-3.5 w-3.5" />
                                    )}
                                  </div>
                                </button>

                                {isTechExpanded && (
                                  <div className="p-3.5 border-t border-slate-100 bg-slate-50/50 space-y-3 text-xs">
                                    {/* Advanced Metrics Grid */}
                                    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
                                      <div className="rounded border border-slate-200 bg-white p-2 space-y-0.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold">
                                          Raw Predictions
                                        </div>
                                        <div className="text-[11px] font-mono text-slate-800 truncate">
                                          {Object.entries(t.original_classes || {})
                                            .map(([cls, cnt]) => `${cls}: ${cnt}`)
                                            .join(', ')}
                                        </div>
                                      </div>

                                      <div className="rounded border border-slate-200 bg-white p-2 space-y-0.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold">
                                          Observed Frames
                                        </div>
                                        <div className="text-[11px] font-mono text-slate-800">
                                          f.{t.first_frame}–f.{t.last_frame} ({t.lifespan_frames} frames)
                                        </div>
                                      </div>

                                      <div className="rounded border border-slate-200 bg-white p-2 space-y-0.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold">
                                          Avg Confidence
                                        </div>
                                        <div className="text-[11px] font-mono text-slate-800">
                                          {(t.average_confidence * 100).toFixed(1)}%
                                        </div>
                                      </div>

                                      <div className="rounded border border-slate-200 bg-white p-2 space-y-0.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold">
                                          Cumulative Disp.
                                        </div>
                                        <div className="text-[11px] font-mono text-slate-800">
                                          {formatDisplacement(motion?.cumulative_pixel_displacement)}
                                        </div>
                                      </div>

                                      <div className="rounded border border-slate-200 bg-white p-2 space-y-0.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold">
                                          Net Disp.
                                        </div>
                                        <div className="text-[11px] font-mono text-slate-800">
                                          {formatDisplacement(motion?.net_pixel_displacement)}
                                        </div>
                                      </div>

                                      <div className="rounded border border-slate-200 bg-white p-2 space-y-0.5">
                                        <div className="text-[10px] text-slate-500 uppercase font-semibold">
                                          Peak Rate
                                        </div>
                                        <div className="text-[11px] font-mono text-slate-800">
                                          {formatPxSpeed(motion?.max_pixel_speed_px_per_sec, isStationary)}
                                        </div>
                                      </div>
                                    </div>

                                    {/* Timing & Physical Speed Limitation Note */}
                                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 p-2.5 rounded border border-amber-200 bg-amber-50/60 text-[11px] text-amber-900">
                                      <div className="flex items-center gap-1.5 font-medium">
                                        <ShieldAlert className="h-3.5 w-3.5 text-amber-600 shrink-0" />
                                        <span>
                                          Physical Speed: {motion?.is_physical_speed_available ? 'Available' : 'Uncalibrated (px/s only)'}
                                        </span>
                                      </div>
                                      {motion?.first_observed_time_seconds != null &&
                                        motion?.last_observed_time_seconds != null && (
                                          <span className="text-slate-600 font-mono text-[10px]">
                                            Timestamp: {motion.first_observed_time_seconds.toFixed(2)}s – {motion.last_observed_time_seconds.toFixed(2)}s
                                          </span>
                                        )}
                                    </div>

                                    {/* Consecutive Frame Displacements Table */}
                                    {motion?.step_displacements &&
                                      motion.step_displacements.length > 0 && (
                                        <div className="rounded border border-slate-200 bg-white p-2.5 space-y-1.5">
                                          <div className="flex items-center justify-between text-xs font-bold text-slate-800">
                                            <span>
                                              Consecutive Frame Displacements ({motion.step_displacements.length} Steps)
                                            </span>
                                            <span className="text-[10px] text-slate-500 font-normal">
                                              Euclidean center-point displacement
                                            </span>
                                          </div>
                                          <div className="overflow-x-auto max-h-40 rounded border border-slate-200">
                                            <table className="w-full text-left text-[11px] border-collapse">
                                              <thead>
                                                <tr className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                                                  <th className="py-1 px-2.5">Frame Interval</th>
                                                  <th className="py-1 px-2.5">Timestamp Span</th>
                                                  <th className="py-1 px-2.5 text-right">Δ Time (s)</th>
                                                  <th className="py-1 px-2.5 text-right">Displacement</th>
                                                  <th className="py-1 px-2.5 text-right">Rate</th>
                                                </tr>
                                              </thead>
                                              <tbody className="divide-y divide-slate-100 font-mono text-[10px]">
                                                {motion.step_displacements.map((step, sIdx) => (
                                                  <tr key={sIdx} className="hover:bg-slate-50/50">
                                                    <td className="py-1 px-2.5 text-slate-800">
                                                      f.{step.from_frame} → f.{step.to_frame}
                                                    </td>
                                                    <td className="py-1 px-2.5 text-slate-600">
                                                      {step.from_timestamp_seconds.toFixed(2)}s → {step.to_timestamp_seconds.toFixed(2)}s
                                                    </td>
                                                    <td className="py-1 px-2.5 text-right text-slate-700">
                                                      {step.elapsed_seconds.toFixed(3)}s
                                                    </td>
                                                    <td className="py-1 px-2.5 text-right font-semibold text-slate-900">
                                                      {step.displacement_pixels.toFixed(1)} px
                                                    </td>
                                                    <td className="py-1 px-2.5 text-right text-blue-700 font-semibold">
                                                      {step.pixel_speed_px_per_sec.toFixed(1)} px/s
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

          {/* Simplified Vehicle Label Corrections Accordion (Temporal Majority Voting) */}
          {reclassifiedDetections && reclassifiedDetections.length > 0 && (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 overflow-hidden">
              <button
                type="button"
                onClick={() => setShowCorrections(!showCorrections)}
                aria-expanded={showCorrections}
                className="w-full flex items-center justify-between p-3 sm:px-4 text-xs font-semibold text-indigo-950 hover:bg-indigo-100/40 transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-indigo-600 shrink-0" />
                  <span className="font-bold">
                    Vehicle Label Corrections ({reclassifiedDetections.length})
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-indigo-700 text-[11px]">
                  <span>{showCorrections ? 'Hide corrections' : 'Show corrections'}</span>
                  {showCorrections ? (
                    <ChevronUp className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5" />
                  )}
                </div>
              </button>

              <div className="px-3 sm:px-4 pb-2.5 text-xs text-indigo-900/90 leading-relaxed">
                The system uses the most frequent predicted vehicle type within a track to reduce inconsistent labels across frames.
              </div>

              {showCorrections && (
                <div className="p-3 sm:p-4 pt-1 border-t border-indigo-100 bg-white space-y-2">
                  <div className="overflow-x-auto rounded-lg border border-indigo-100">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-indigo-50/70 text-indigo-900 font-bold text-[10px] uppercase">
                          <th className="py-2 px-3">Frame #</th>
                          <th className="py-2 px-3">Track ID</th>
                          <th className="py-2 px-3">Original Predicted Label</th>
                          <th className="py-2 px-3">Final Label</th>
                          <th className="py-2 px-3 text-right">Confidence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-indigo-50">
                        {reclassifiedDetections.map((item, idx) => (
                          <tr key={idx} className="hover:bg-indigo-50/30">
                            <td className="py-1.5 px-3 font-mono text-slate-700">
                              f.{item.frame_number}
                            </td>
                            <td className="py-1.5 px-3 font-mono font-bold text-slate-900">
                              #{item.track_id}
                            </td>
                            <td className="py-1.5 px-3">
                              <span className="line-through text-rose-600 font-medium mr-1">
                                {item.raw_class}
                              </span>
                            </td>
                            <td className="py-1.5 px-3">
                              <span className="font-bold text-emerald-700 capitalize">
                                {item.assigned_class}
                              </span>
                            </td>
                            <td className="py-1.5 px-3 text-right font-mono text-slate-700">
                              {(item.confidence * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[11px] text-slate-500 italic pt-1">
                    Note: Label correction applies majority voting to reduce flicker on a single track. It does not resolve tracking ID switches or guarantee ground-truth classification.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
