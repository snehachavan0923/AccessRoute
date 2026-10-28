import { Car, Bus, Bike, Truck } from 'lucide-react'

const CLASS_CONFIG = {
  car: {
    label: 'Cars',
    icon: Car,
    colorBadge: 'bg-blue-100 text-blue-800 border-blue-200',
    barColor: 'bg-blue-500',
    dotColor: 'bg-blue-500',
  },
  bus: {
    label: 'Buses',
    icon: Bus,
    colorBadge: 'bg-amber-100 text-amber-800 border-amber-200',
    barColor: 'bg-amber-500',
    dotColor: 'bg-amber-500',
  },
  motorcycle: {
    label: 'Motorcycles',
    icon: Bike,
    colorBadge: 'bg-purple-100 text-purple-800 border-purple-200',
    barColor: 'bg-purple-500',
    dotColor: 'bg-purple-500',
  },
  truck: {
    label: 'Trucks',
    icon: Truck,
    colorBadge: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    barColor: 'bg-emerald-500',
    dotColor: 'bg-emerald-500',
  },
}

export default function VisionClassBreakdown({ report }) {
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
            Per-class comparison of raw detection events versus post-processed tracking IDs
          </p>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-slate-400" />
            Detection Events ≠ Physical Vehicle Count
          </span>
        </div>
      </div>

      {/* Visual Composition Bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-[11px] font-semibold text-slate-600">
          <span>Processed Detection Distribution</span>
          <span>{report.processed_total_detections.toLocaleString()} total events</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-100 flex">
          {targetClasses.map((cls) => {
            const count = processedDetections[cls] || 0
            if (count === 0) return null
            const pct = (count / totalProcessedDets) * 100
            const cfg = CLASS_CONFIG[cls]
            return (
              <div
                key={cls}
                style={{ width: `${pct}%` }}
                className={`${cfg.barColor} transition-all duration-300 relative group`}
                title={`${cfg.label}: ${count} (${pct.toFixed(1)}%)`}
              />
            )
          })}
        </div>
      </div>

      {/* Grid of Class Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {targetClasses.map((cls) => {
          const cfg = CLASS_CONFIG[cls]
          const Icon = cfg.icon

          const rawDetCount = rawDetections[cls] || 0
          const procDetCount = processedDetections[cls] || 0
          const rawTrackCount = rawTracked[cls] || 0
          const procTrackCount = processedTracked[cls] || 0

          const detDiff = procDetCount - rawDetCount

          return (
            <div
              key={cls}
              className="rounded-xl border border-slate-100 bg-slate-50/70 p-3.5 space-y-2.5 transition-all hover:bg-slate-50"
            >
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-lg border ${cfg.colorBadge}`}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-bold text-slate-900">
                    {cfg.label}
                  </span>
                </div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  COCO #{cls === 'car' ? 2 : cls === 'motorcycle' ? 3 : cls === 'bus' ? 5 : 7}
                </span>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-2 pt-1">
                {/* Detection Events */}
                <div className="rounded-lg bg-white p-2 border border-slate-200/60 shadow-2xs">
                  <span className="text-[10px] font-bold uppercase text-slate-400 block">
                    Detections
                  </span>
                  <div className="flex items-baseline gap-1 mt-0.5">
                    <span className="text-base font-extrabold text-slate-900">
                      {procDetCount.toLocaleString()}
                    </span>
                    {detDiff !== 0 && (
                      <span
                        className={`text-[10px] font-bold ${
                          detDiff > 0 ? 'text-emerald-600' : 'text-amber-600'
                        }`}
                        title="Change after majority voting and lifespan filtering"
                      >
                        {detDiff > 0 ? `+${detDiff}` : detDiff}
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-slate-500 block truncate">
                    Raw: {rawDetCount.toLocaleString()}
                  </span>
                </div>

                {/* Tracked IDs */}
                <div className="rounded-lg bg-white p-2 border border-slate-200/60 shadow-2xs">
                  <span className="text-[10px] font-bold uppercase text-slate-400 block">
                    Tracked IDs
                  </span>
                  <div className="flex items-baseline gap-1 mt-0.5">
                    <span className="text-base font-extrabold text-slate-900">
                      {procTrackCount}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-500 block truncate">
                    Raw IDs: {rawTrackCount}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
