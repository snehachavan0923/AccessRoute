import { useState } from 'react'
import { Info, ChevronDown, ChevronUp, AlertCircle, ShieldAlert, Video, Split, Layers, Gauge } from 'lucide-react'

export default function VisionDisclaimer({ disclaimer }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div
      role="region"
      aria-label="About These Results"
      className="rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-xs transition-all"
    >
      {/* Compact default information box */}
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
            <span>{isExpanded ? 'Hide Technical Limitations' : 'Technical Limitations'}</span>
            {isExpanded ? (
              <ChevronUp className="h-3.5 w-3.5 text-slate-500" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
            )}
          </button>
        </div>

        {/* Exact user-specified default explanation */}
        <p className="text-xs sm:text-[13px] text-slate-700 leading-relaxed font-normal">
          These results are AI estimates. Vehicle detections and tracking may contain errors. Movement is measured in image pixels, not real-world speed. This analysis does not independently verify traffic congestion or pedestrian accessibility.
        </p>
      </div>

      {/* Expandable Technical Limitations (hidden by default) */}
      {isExpanded && (
        <div className="px-4 pb-4 sm:px-5 sm:pb-5 pt-3.5 border-t border-slate-100 bg-slate-50/70 space-y-3.5">
          <div className="flex items-center justify-between">
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Detailed Technical Limitations
            </div>
            <span className="text-[11px] text-slate-400">
              Methodology &amp; Scope Constraints
            </span>
          </div>

          {disclaimer && (
            <p className="text-xs text-slate-700 leading-relaxed bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs">
              {disclaimer}
            </p>
          )}

          {/* 6 Key Technical Limitation Areas */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-xs text-slate-700">
            {/* 1. Detection Events vs. Tracked IDs */}
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-slate-900">
                <Video className="h-3.5 w-3.5 text-blue-600" />
                <span>Detection Events vs. Tracked IDs</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                A single vehicle observed over 30 frames generates 30 detection events. Distinct tracking IDs represent temporal trajectories, but cannot be guaranteed to equal the physical number of unique vehicles entering the scene.
              </p>
            </div>

            {/* 2. Temporal Majority Voting */}
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-slate-900">
                <Layers className="h-3.5 w-3.5 text-indigo-600" />
                <span>Temporal Majority Voting</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Within each track, majority voting reduces classification flicker across frames. It stabilizes labels on noisy predictions but does not guarantee ground-truth vehicle classification accuracy.
              </p>
            </div>

            {/* 3. Tracking Fragmentation & ID Switches */}
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-slate-900">
                <Split className="h-3.5 w-3.5 text-purple-600" />
                <span>Tracking Fragmentation &amp; ID Switches</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Temporary occlusions, crossing paths, or objects leaving and re-entering the frame can cause ByteTrack to assign new tracking IDs (fragmentation) or swap IDs between vehicles.
              </p>
            </div>

            {/* 4. Image-Space Motion Limitations */}
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-slate-900">
                <Gauge className="h-3.5 w-3.5 text-amber-600" />
                <span>Image-Space Motion Limitations</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Displacement and rates are measured in 2D sensor pixels (px, px/s). Perspective foreshortening causes distant vehicles to traverse fewer pixels per meter than foreground vehicles at the identical physical speed.
              </p>
            </div>

            {/* 5. Camera Calibration Requirements */}
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-slate-900">
                <AlertCircle className="h-3.5 w-3.5 text-rose-600" />
                <span>Camera Calibration Requirements</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Estimating real-world physical velocity (km/h) requires full camera-to-road homography (<em>H</em>), lens distortion correction, and surveyed spatial landmarks. Without calibration, physical speeds cannot be determined.
              </p>
            </div>

            {/* 6. Traffic Congestion & Accessibility Scope */}
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
              <div className="flex items-center gap-1.5 font-bold text-slate-900">
                <ShieldAlert className="h-3.5 w-3.5 text-emerald-600" />
                <span>Congestion &amp; Accessibility Scope</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                This computer vision module does not detect sidewalk barriers, stairs, kerbs, surface hazards, or pedestrian accessibility. These video metrics do not alter AccessRoute route recommendations.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

