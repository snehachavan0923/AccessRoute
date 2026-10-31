import { useState } from 'react'
import { Car, Bus, Bike, Truck, ChevronDown, ChevronUp } from 'lucide-react'

const CLASS_CONFIG = {
  car: {
    label: 'Cars',
    icon: Car,
    colorBadge: 'bg-blue-100 text-blue-800 border-blue-200',
    barColor: 'bg-blue-500',
  },
  bus: {
    label: 'Buses',
    icon: Bus,
    colorBadge: 'bg-amber-100 text-amber-800 border-amber-200',
    barColor: 'bg-amber-500',
  },
  motorcycle: {
    label: 'Motorcycles',
    icon: Bike,
    colorBadge: 'bg-purple-100 text-purple-800 border-purple-200',
    barColor: 'bg-purple-500',
  },
  truck: {
    label: 'Trucks',
    icon: Truck,
    colorBadge: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    barColor: 'bg-emerald-500',
  },
}

export default function VisionClassBreakdown({ report }) {
  const [showRawDetails, setShowRawDetails] = useState(false)
  if (!report) return null

  const targetClasses = ['car', 'bus', 'motorcycle', 'truck']
  const rawDetections = report.raw_detection_events_by_class || {}
  const processedDetections = report.processed_detection_events_by_class || {}
  const rawTracked = report.raw_tracked_ids_by_class || {}
  const processedTracked = report.processed_tracked_ids_by_class || {}

  const totalProcessedDets = Math.max(1, report.processed_total_detections || 1)

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 sm:p-5 shadow-xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900">
            Vehicle Category Breakdown
          </h3>
          <p className="text-xs text-slate-500">
            Observed vehicles categorized by vehicle type across video frames.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowRawDetails(!showRawDetails)}
          className="self-start sm:self-auto flex items-center gap-1 text-[11px] font-semibold text-slate-500 hover:text-slate-800 bg-slate-100 hover:bg-slate-200/80 px-2.5 py-1 rounded-lg transition-colors cursor-pointer"
        >
          <span>{showRawDetails ? 'Hide Raw Details' : 'Show Raw Counts'}</span>
          {showRawDetails ? (
            <ChevronUp className="h-3 w-3" />
          ) : (
            <ChevronDown className="h-3 w-3" />
          )}
        </button>
      </div>

      {/* Visual Composition Bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[11px] font-semibold text-slate-600">
          <span>Detection Distribution</span>
          <span>{report.processed_total_detections.toLocaleString()} total detections</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100 flex">
          {targetClasses.map((cls) => {
            const count = processedDetections[cls] || 0
            if (count === 0) return null
            const pct = (count / totalProcessedDets) * 100
            const cfg = CLASS_CONFIG[cls]
            return (
              <div
                key={cls}
                style={{ width: `${pct}%` }}
                className={`${cfg.barColor} transition-all duration-300`}
                title={`${cfg.label}: ${count} (${pct.toFixed(1)}%)`}
              />
            )
          })}
        </div>
      </div>

      {/* Grid of Class Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {targetClasses.map((cls) => {
          const cfg = CLASS_CONFIG[cls]
          const Icon = cfg.icon

          const procDetCount = processedDetections[cls] || 0
          const procTrackCount = processedTracked[cls] || 0
          const rawDetCount = rawDetections[cls] || 0
          const rawTrackCount = rawTracked[cls] || 0

          return (
            <div
              key={cls}
              className="rounded-xl border border-slate-100 bg-slate-50/70 p-3.5 space-y-2 transition-all hover:bg-slate-50"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-lg border ${cfg.colorBadge}`}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-bold text-slate-900">
                    {cfg.label}
                  </span>
                </div>
              </div>

              <div className="pt-0.5">
                <div className="text-xl font-extrabold text-slate-900">
                  {procTrackCount} <span className="text-xs font-semibold text-slate-500 font-sans">tracked</span>
                </div>
                <div className="text-[11px] text-slate-600 mt-0.5">
                  {procDetCount.toLocaleString()} detections
                </div>
              </div>

              {showRawDetails && (
                <div className="pt-2 border-t border-slate-200/80 text-[10px] text-slate-500 space-y-0.5 font-mono">
                  <div>Raw IDs: {rawTrackCount}</div>
                  <div>Raw Detections: {rawDetCount.toLocaleString()}</div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

