import { useState } from 'react'
import { Info, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react'

export default function VisionDisclaimer({ disclaimer }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div
      role="region"
      aria-label="About These Results"
      className="rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-xs transition-all"
    >
      {/* Header bar and default summary text */}
      <div className="p-4 sm:p-5 space-y-2.5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
              <Info className="h-4 w-4" />
            </span>
            <h3 className="text-sm font-bold text-slate-900">
              About These Results
            </h3>
          </div>

          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-expanded={isExpanded}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200/80 rounded-lg transition-colors cursor-pointer"
          >
            <span>{isExpanded ? 'Hide technical details' : 'Technical details'}</span>
            {isExpanded ? (
              <ChevronUp className="h-3.5 w-3.5 text-slate-500" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
            )}
          </button>
        </div>

        {/* Short text displayed by default */}
        <p className="text-xs sm:text-[13px] text-slate-600 leading-relaxed">
          Vehicle detection and tracking results are experimental. Detection counts are not actual vehicle counts, and tracking IDs may change. Traffic congestion and pedestrian accessibility are not yet verified by this analysis.
        </p>
      </div>

      {/* Collapsible detailed technical explanations */}
      {isExpanded && (
        <div className="px-4 pb-4 sm:px-5 sm:pb-5 pt-3 border-t border-slate-100 bg-slate-50/70 space-y-3">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
            Detailed Technical Explanations
          </div>

          <p className="text-xs text-slate-700 leading-relaxed bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs">
            {disclaimer ||
              'Detection events measure per-frame bounding boxes; tracker IDs measure spatial trajectories. Neither metric represents verified ground-truth physical vehicle counts.'}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs text-slate-700">
            <div className="flex items-start gap-2.5 bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs">
              <AlertCircle className="h-4 w-4 shrink-0 text-blue-600 mt-0.5" />
              <div className="space-y-0.5">
                <strong className="block font-semibold text-slate-900 text-xs">
                  Detection Events vs. Tracked IDs
                </strong>
                <span className="text-[11px] text-slate-600 leading-snug block">
                  A vehicle present across 30 frames generates 30 detection events. Distinct tracker IDs track spatial trajectories across time, but remain subject to occlusions, ID switches, and trajectory fragmentation.
                </span>
              </div>
            </div>

            <div className="flex items-start gap-2.5 bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs">
              <AlertCircle className="h-4 w-4 shrink-0 text-indigo-600 mt-0.5" />
              <div className="space-y-0.5">
                <strong className="block font-semibold text-slate-900 text-xs">
                  Post-Processing & Accessibility Limitations
                </strong>
                <span className="text-[11px] text-slate-600 leading-snug block">
                  Temporal majority voting smooths oscillating raw class labels on a single track. It does not certify street accessibility barriers, sidewalk obstacles, or ground-truth traffic congestion.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
