import { useState } from 'react'
import { Play, Sparkles, Video, SplitSquareVertical, ExternalLink } from 'lucide-react'
import { getFullMediaUrl } from '../../services/vision'

export default function VisionVideoPlayer({
  sourceVideoUrl,
  annotatedVideoUrl,
  metadata,
  processedFrames,
}) {
  const [activeView, setActiveView] = useState('annotated') // 'annotated' | 'original' | 'split'

  const fullSourceUrl = getFullMediaUrl(sourceVideoUrl)
  const fullAnnotatedUrl = getFullMediaUrl(annotatedVideoUrl)

  const hasAnnotated = Boolean(annotatedVideoUrl)

  return (
    <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-xs">
      {/* Header & View Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 p-3.5 sm:px-5 sm:py-3 border-b border-slate-100 bg-slate-50/70">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600 text-white shadow-2xs">
            <Video className="h-3.5 w-3.5" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-900 leading-none">
              Video Playback & Visual Annotations
            </h3>
            <span className="text-[11px] text-slate-500">
              {metadata?.video_name || 'Traffic Footage Stream'}
            </span>
          </div>
        </div>

        {/* View Toggle Controls */}
        <div className="flex items-center gap-1.5 self-start sm:self-auto bg-slate-200/70 p-1 rounded-xl text-xs font-medium">
          {hasAnnotated && (
            <button
              type="button"
              onClick={() => setActiveView('annotated')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all ${
                activeView === 'annotated'
                  ? 'bg-white text-blue-700 font-semibold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sparkles className="h-3.5 w-3.5 text-blue-600" />
              <span>Annotated</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => setActiveView('original')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all ${
              activeView === 'original' || !hasAnnotated
                ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Play className="h-3.5 w-3.5 text-slate-500" />
            <span>Original</span>
          </button>

          {hasAnnotated && (
            <button
              type="button"
              onClick={() => setActiveView('split')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all ${
                activeView === 'split'
                  ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="View both original and annotated video side-by-side"
            >
              <SplitSquareVertical className="h-3.5 w-3.5 text-slate-500" />
              <span className="hidden md:inline">Side-by-Side</span>
            </button>
          )}
        </div>
      </div>

      {/* Video Display Area */}
      <div className="p-3 sm:p-5 bg-slate-950/95 flex flex-col items-center justify-center min-h-[300px]">
        {activeView === 'split' && hasAnnotated ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-5xl">
            {/* Original Stream */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400 px-1">
                <span>Original Source Footage</span>
                <span className="text-slate-500">{metadata?.width || 640}x{metadata?.height || 360}</span>
              </div>
              <div className="overflow-hidden rounded-xl border border-slate-800 bg-black aspect-video flex items-center justify-center">
                <video
                  src={fullSourceUrl}
                  controls
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            {/* Annotated Stream */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-semibold text-blue-400 px-1">
                <span className="flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  <span>YOLOv11n + ByteTrack Annotated</span>
                </span>
                <span className="text-blue-300/80">{processedFrames} frames evaluated</span>
              </div>
              <div className="overflow-hidden rounded-xl border border-blue-950 bg-black aspect-video flex items-center justify-center">
                <video
                  src={fullAnnotatedUrl}
                  controls
                  autoPlay
                  loop
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="w-full max-w-3xl space-y-2">
            <div className="overflow-hidden rounded-xl border border-slate-800 bg-black aspect-video flex items-center justify-center shadow-lg">
              {activeView === 'annotated' && hasAnnotated ? (
                <video
                  key="annotated"
                  src={fullAnnotatedUrl}
                  controls
                  autoPlay
                  loop
                  playsInline
                  className="w-full h-full object-contain"
                />
              ) : (
                <video
                  key="original"
                  src={fullSourceUrl}
                  controls
                  playsInline
                  className="w-full h-full object-contain"
                />
              )}
            </div>

            {/* Status caption under video */}
            <div className="flex items-center justify-between px-1 text-[11px] text-slate-400">
              <span className="flex items-center gap-1.5">
                <span
                  className={`h-2 w-2 rounded-full ${
                    activeView === 'annotated' && hasAnnotated
                      ? 'bg-blue-400 animate-pulse'
                      : 'bg-emerald-400'
                  }`}
                />
                <span>
                  {activeView === 'annotated' && hasAnnotated
                    ? 'Showing YOLO detection bounding boxes, tracking IDs, and majority-voted labels'
                    : 'Showing unannotated source video'}
                </span>
              </span>

              {hasAnnotated && (
                <a
                  href={fullAnnotatedUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-blue-400 hover:text-blue-300 hover:underline"
                >
                  <span>Open Video in New Tab</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Metadata Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-3 sm:px-5 border-t border-slate-100 bg-slate-50/50 text-[11px] text-slate-600">
        <div>
          <span className="text-slate-400 block text-[10px] uppercase font-bold">
            Dimensions
          </span>
          <span className="font-semibold text-slate-800">
            {metadata?.width || 640} x {metadata?.height || 360} px
          </span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px] uppercase font-bold">
            Framerate
          </span>
          <span className="font-semibold text-slate-800">
            {metadata?.video_fps ? `${metadata.video_fps.toFixed(2)} FPS` : '29.97 FPS'}
          </span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px] uppercase font-bold">
            Video Duration
          </span>
          <span className="font-semibold text-slate-800">
            {metadata?.video_duration_seconds
              ? `${metadata.video_duration_seconds.toFixed(1)}s (${metadata.total_frames_in_video} frames)`
              : '10.0s'}
          </span>
        </div>
        <div>
          <span className="text-slate-400 block text-[10px] uppercase font-bold">
            Evaluated Span
          </span>
          <span className="font-semibold text-blue-700">
            {processedFrames} frames evaluated
          </span>
        </div>
      </div>
    </div>
  )
}
