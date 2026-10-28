import { Activity, Layers, Filter, CheckCircle2 } from 'lucide-react'

export default function VisionMetricsCards({ report, processingTime }) {
  if (!report) return null

  const cards = [
    {
      title: 'Total Detection Events',
      value: report.processed_total_detections,
      secondary:
        report.raw_total_detections !== report.processed_total_detections
          ? `Raw: ${report.raw_total_detections.toLocaleString()} events`
          : 'All detections retained',
      description: 'Cumulative per-frame bounding boxes observed across evaluated frames.',
      icon: Activity,
      color: 'blue',
      badge: `${report.processed_frames} frames`,
    },
    {
      title: 'Distinct Tracked IDs',
      value: report.processed_total_tracked_ids,
      secondary:
        report.raw_total_tracked_ids !== report.processed_total_tracked_ids
          ? `Raw: ${report.raw_total_tracked_ids} tracker IDs`
          : 'No IDs pruned',
      description: 'ByteTrack spatial trajectories meeting minimum lifespan criteria.',
      icon: Layers,
      color: 'emerald',
      badge: 'Trajectory tracking',
    },
    {
      title: 'Filtered Transient Tracks',
      value: report.filtered_tracks_count,
      secondary: `${report.filtered_detections_count} transient boxes pruned`,
      description: `Tracks active for fewer than ${report.min_track_lifespan_frames} frames pruned as transient noise.`,
      icon: Filter,
      color: 'amber',
      badge: `< ${report.min_track_lifespan_frames} frames`,
    },
    {
      title: 'Class-Smoothed Tracks',
      value: report.resolved_class_switches_count,
      secondary:
        (report.reclassified_detections?.length || 0) > 0
          ? `${report.reclassified_detections.length} detections smoothed across ${report.resolved_class_switches_count} track${report.resolved_class_switches_count === 1 ? '' : 's'}`
          : report.raw_class_switches_count > 0
          ? `${report.raw_class_switches_count} tracks with raw class switches`
          : 'Stable labels (0 oscillations)',
      description: 'Retained vehicle tracks whose oscillating raw class labels were resolved to a single category by temporal majority voting.',
      icon: CheckCircle2,
      color: 'indigo',
      badge:
        (report.reclassified_detections?.length || 0) > 0
          ? `${report.reclassified_detections.length} detections`
          : 'Majority voted',
    },
  ]

  const colorStyles = {
    blue: {
      bg: 'bg-blue-50/80',
      border: 'border-blue-200/80',
      text: 'text-blue-900',
      iconBg: 'bg-blue-600',
      badgeBg: 'bg-blue-100 text-blue-700',
    },
    emerald: {
      bg: 'bg-emerald-50/80',
      border: 'border-emerald-200/80',
      text: 'text-emerald-900',
      iconBg: 'bg-emerald-600',
      badgeBg: 'bg-emerald-100 text-emerald-700',
    },
    amber: {
      bg: 'bg-amber-50/80',
      border: 'border-amber-200/80',
      text: 'text-amber-900',
      iconBg: 'bg-amber-600',
      badgeBg: 'bg-amber-100 text-amber-700',
    },
    indigo: {
      bg: 'bg-indigo-50/80',
      border: 'border-indigo-200/80',
      text: 'text-indigo-900',
      iconBg: 'bg-indigo-600',
      badgeBg: 'bg-indigo-100 text-indigo-700',
    },
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Core Vision Analytics & Tracking Metrics
        </h3>
        {processingTime !== undefined && (
          <span className="text-[11px] font-medium text-slate-500">
            Inference & pipeline completed in{' '}
            <strong className="text-slate-700">{processingTime}s</strong>
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {cards.map((card, idx) => {
          const style = colorStyles[card.color]
          const Icon = card.icon
          return (
            <div
              key={idx}
              className={`rounded-2xl border ${style.border} ${style.bg} p-4 shadow-xs flex flex-col justify-between transition-all hover:shadow-sm`}
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-xl ${style.iconBg} text-white shadow-2xs`}
                  >
                    <Icon className="h-4 w-4" />
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide ${style.badgeBg}`}
                  >
                    {card.badge}
                  </span>
                </div>

                <div className="text-2xl font-black tracking-tight text-slate-900">
                  {typeof card.value === 'number'
                    ? card.value.toLocaleString()
                    : card.value}
                </div>

                <div className="text-xs font-bold text-slate-800 mt-0.5">
                  {card.title}
                </div>

                <div className="text-[11px] font-semibold text-slate-600 mt-1">
                  {card.secondary}
                </div>
              </div>

              <p className="mt-2.5 pt-2 border-t border-slate-200/60 text-[11px] leading-snug text-slate-500">
                {card.description}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
