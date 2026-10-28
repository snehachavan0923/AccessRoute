import { useState } from 'react'
import { Search, ChevronDown, ChevronUp, Check, AlertCircle } from 'lucide-react'

export default function VisionTrackletTable({ tracklets, reclassifiedDetections, minLifespan }) {
  const [classFilter, setClassFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all') // 'all' | 'retained' | 'filtered'
  const [searchTerm, setSearchTerm] = useState('')
  const [isExpanded, setIsExpanded] = useState(true)

  if (!tracklets || tracklets.length === 0) return null

  const retainedCount = tracklets.filter((t) => !t.is_filtered).length
  const filteredCount = tracklets.filter((t) => t.is_filtered).length

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
    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      const matchesId = String(t.track_id).includes(term)
      const matchesClass = t.assigned_class.toLowerCase().includes(term)
      if (!matchesId && !matchesClass) return false
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
            Tracklet Trajectory Details ({retainedCount} Retained, {filteredCount} Filtered / {tracklets.length} Total Tracks)
          </h3>
          <span className="rounded-full bg-slate-200/80 px-2 py-0.5 text-[10px] font-bold text-slate-700">
            ByteTrack + Post-Processing
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
            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 sm:p-5 space-y-4">
          {/* Controls: Search & Category Filters */}
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
                        ? 'bg-white text-blue-700 shadow-2xs'
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
            </div>

            {/* Search Input */}
            <div className="relative min-w-[180px]">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search track ID or class…"
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
                  <th className="py-2.5 px-3">Raw Predictions</th>
                  <th className="py-2.5 px-3 text-right">Detections</th>
                  <th className="py-2.5 px-3 text-right">Observed Frames</th>
                  <th className="py-2.5 px-3 text-right">Lifespan Span</th>
                  <th className="py-2.5 px-3 text-right">Avg Confidence</th>
                  <th className="py-2.5 px-3 text-center">Post-Proc Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredTracklets.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-slate-400">
                      No tracklets match the selected filters.
                    </td>
                  </tr>
                ) : (
                  filteredTracklets.map((t) => (
                    <tr
                      key={t.track_id}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        t.is_filtered ? 'opacity-60 bg-slate-50/40' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 font-mono font-bold text-slate-900">
                        #{t.track_id}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-bold capitalize bg-blue-50 text-blue-700 border border-blue-200/60">
                          {t.assigned_class}
                        </span>
                      </td>
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
                      <td className="py-2.5 px-3 text-right font-semibold text-slate-800">
                        {t.detection_count}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-700">
                        {t.observed_frame_count}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-500 font-mono text-[11px]">
                        f.{t.first_frame}–f.{t.last_frame} ({t.lifespan_frames} frames)
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono font-semibold text-slate-700">
                        {(t.average_confidence * 100).toFixed(1)}%
                      </td>
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
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Reclassified Detections (Temporal Majority Voting Results) */}
          {reclassifiedDetections && reclassifiedDetections.length > 0 && (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-indigo-900">
                  Intra-Tracklet Label Corrections ({reclassifiedDetections.length} Detections Smoothed)
                </span>
                <span className="text-[10px] text-indigo-600 font-medium">
                  Temporal Majority Voting
                </span>
              </div>
              <p className="text-[11px] text-indigo-800 leading-snug">
                The following individual detections had an unstable class prediction on an active track that was smoothed to the winning majority category:
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
                        <td className="py-1.5 px-2.5 font-mono text-slate-800">f.{item.frame_number}</td>
                        <td className="py-1.5 px-2.5 text-slate-600 font-mono">{item.timestamp_seconds.toFixed(2)}s</td>
                        <td className="py-1.5 px-2.5 font-mono font-bold text-slate-900">#{item.track_id}</td>
                        <td className="py-1.5 px-2.5">
                          <span className="line-through text-rose-600 font-medium mr-1.5">{item.raw_class}</span>
                        </td>
                        <td className="py-1.5 px-2.5">
                          <span className="font-bold text-emerald-700">{item.assigned_class}</span>
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
