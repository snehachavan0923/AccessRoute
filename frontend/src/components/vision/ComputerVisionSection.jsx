import { useEffect, useRef, useState } from 'react'
import {
  Sparkles,
  Play,
  UploadCloud,
  FileVideo,
  Sliders,
  Loader2,
  AlertCircle,
  Activity,
} from 'lucide-react'
import {
  analyzeVideo,
  getVisionStatus,
  uploadVideo,
} from '../../services/vision'
import VisionVideoPlayer from './VisionVideoPlayer'
import VisionMetricsCards from './VisionMetricsCards'
import VisionClassBreakdown from './VisionClassBreakdown'
import VisionTrackletTable from './VisionTrackletTable'
import VisionDisclaimer from './VisionDisclaimer'

export default function ComputerVisionSection() {
  const [status, setStatus] = useState(null)
  const [statusLoading, setStatusLoading] = useState(true)

  // Video Selection
  const [selectedVideoSource, setSelectedVideoSource] = useState('sample') // 'sample' | 'upload'
  const [uploadedVideo, setUploadedVideo] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')

  // Analysis Parameters
  const [frameMode, setFrameMode] = useState('30') // '30' | 'full' | 'custom'
  const [customFrames, setCustomFrames] = useState(60)
  const [confThreshold, setConfThreshold] = useState(0.5)
  const [iouThreshold, setIouThreshold] = useState(0.3)
  const [minLifespan, setMinLifespan] = useState(5)
  const [generateAnnotated, setGenerateAnnotated] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Execution State
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgressText, setAnalysisProgressText] = useState('')
  const [analysisError, setAnalysisError] = useState('')
  const [analysisResult, setAnalysisResult] = useState(null)

  const abortControllerRef = useRef(null)
  const fileInputRef = useRef(null)

  // Load vision status and available sample videos on mount
  useEffect(() => {
    const controller = new AbortController()
    getVisionStatus(controller.signal)
      .then((data) => {
        setStatus(data)
        setStatusLoading(false)
      })
      .catch((err) => {
        console.error('Failed to load vision status:', err)
        setStatusLoading(false)
      })

    return () => controller.abort()
  }, [])

  function handleFileSelected(e) {
    const file = e.target.files?.[0]
    if (!file) return

    setUploadError('')
    setIsUploading(true)

    uploadVideo(file)
      .then((res) => {
        setUploadedVideo(res)
        setSelectedVideoSource('upload')
        setAnalysisResult(null)
        setAnalysisError('')
        setIsUploading(false)
      })
      .catch((err) => {
        setUploadError(err.message || 'Failed to upload video')
        setIsUploading(false)
      })
  }

  function handleRunAnalysis() {
    if (isAnalyzing) return

    setAnalysisError('')
    setIsAnalyzing(true)
    setAnalysisProgressText('Submitting video to YOLO detection pipeline…')

    const videoId =
      selectedVideoSource === 'upload' && uploadedVideo
        ? uploadedVideo.video_id
        : 'sample_traffic_video'

    let maxFrames = 30
    if (frameMode === 'full') {
      maxFrames =
        selectedVideoSource === 'upload' && uploadedVideo
          ? uploadedVideo.total_frames
          : status?.sample_videos?.[0]?.total_frames || 301
    } else if (frameMode === 'custom') {
      maxFrames = Math.max(1, Number(customFrames) || 30)
    }

    const payload = {
      videoId,
      maxFrames,
      conf: confThreshold,
      iou: iouThreshold,
      minTrackLifespanFrames: minLifespan,
      generateAnnotatedVideo: generateAnnotated,
    }

    // Step message updates for user feedback
    const progressTimer = setTimeout(() => {
      setAnalysisProgressText('Tracking vehicle trajectories with ByteTrack…')
    }, 1500)

    const postProcTimer = setTimeout(() => {
      setAnalysisProgressText(
        'Post-processing tracklets & rendering annotated video…',
      )
    }, 3500)

    abortControllerRef.current = new AbortController()

    analyzeVideo(payload, abortControllerRef.current.signal)
      .then((res) => {
        clearTimeout(progressTimer)
        clearTimeout(postProcTimer)
        setAnalysisResult(res)
        setIsAnalyzing(false)
        setAnalysisProgressText('')
      })
      .catch((err) => {
        clearTimeout(progressTimer)
        clearTimeout(postProcTimer)
        if (err.name !== 'AbortError') {
          setAnalysisError(err.message || 'Vision analysis failed.')
        }
        setIsAnalyzing(false)
        setAnalysisProgressText('')
      })
  }

  const sampleInfo = status?.sample_videos?.[0]
  const currentVideo =
    selectedVideoSource === 'upload' && uploadedVideo
      ? {
          name: uploadedVideo.filename,
          duration: `${uploadedVideo.duration_seconds}s`,
          frames: uploadedVideo.total_frames,
          resolution: `${uploadedVideo.width}x${uploadedVideo.height}`,
          fps: `${uploadedVideo.fps} FPS`,
          previewUrl: uploadedVideo.video_url,
        }
      : {
          name: sampleInfo?.name || 'Electronic City Traffic Footage',
          duration: sampleInfo ? `${sampleInfo.duration_seconds}s` : '10.0s',
          frames: sampleInfo?.total_frames || 301,
          resolution: sampleInfo ? `${sampleInfo.width}x${sampleInfo.height}` : '640x360',
          fps: sampleInfo ? `${sampleInfo.fps} FPS` : '29.97 FPS',
          previewUrl: sampleInfo?.video_url || '/api/vision/sample-video',
        }

  return (
    <div className="space-y-6 max-w-7xl mx-auto w-full pb-10">
      {/* Header Banner */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1 max-w-2xl">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-blue-600 text-white shadow-2xs">
              <Sparkles className="h-3.5 w-3.5" />
            </span>
            <span className="text-xs font-bold uppercase tracking-wider text-blue-600">
              Computer Vision Pipeline
            </span>
            <span className="rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 text-[10px] font-bold">
              YOLOv11n + ByteTrack
            </span>
          </div>

          <h2 className="text-xl sm:text-2xl font-black tracking-tight text-slate-900">
            Real-Time Vehicle Detection & Tracklet Analysis
          </h2>

          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Run real neural-network inference using local YOLO11n weights and ByteTrack multi-object tracking.
            Evaluate raw detection events, trajectory lifespans, and temporal majority voting on actual traffic footage.
          </p>
        </div>

        {/* Model State Status Badge */}
        <div className="flex flex-col items-start md:items-end gap-1.5 shrink-0">
          <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                status?.weights_available
                  ? 'bg-emerald-500 ring-4 ring-emerald-100'
                  : 'bg-rose-500'
              }`}
            />
            <div className="flex flex-col text-left">
              <span className="font-bold text-slate-800">
                {statusLoading
                  ? 'Checking Model Weights…'
                  : status?.weights_available
                  ? `Weights Loaded: ${status.model_name}`
                  : 'Weights Missing'}
              </span>
              <span className="text-[10px] text-slate-500">
                Target Device: {status?.device?.toUpperCase() || 'CPU'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid: Controls & Source on Left, Player/Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] xl:grid-cols-[410px_1fr] gap-6 items-start">
        {/* Left Column: Video Input & Inference Options */}
        <aside className="space-y-4" aria-label="Computer Vision Analysis Controls">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 sm:p-5 shadow-xs space-y-4">
            <div className="border-b border-slate-100 pb-2.5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                1. Select Video Source
              </h3>
            </div>

            {/* Source Switcher */}
            <div className="grid grid-cols-2 gap-2 p-1 bg-slate-100 rounded-xl text-xs font-semibold">
              <button
                type="button"
                onClick={() => {
                  setSelectedVideoSource('sample')
                  if (analysisResult && analysisResult.video_id !== 'sample_traffic_video') {
                    setAnalysisResult(null)
                    setAnalysisError('')
                  }
                }}
                className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition-all ${
                  selectedVideoSource === 'sample'
                    ? 'bg-white text-blue-700 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <FileVideo className="h-4 w-4" />
                <span>Sample Video</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setSelectedVideoSource('upload')
                  if (analysisResult && analysisResult.video_id === 'sample_traffic_video') {
                    setAnalysisResult(null)
                    setAnalysisError('')
                  }
                }}
                className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition-all ${
                  selectedVideoSource === 'upload'
                    ? 'bg-white text-blue-700 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <UploadCloud className="h-4 w-4" />
                <span>Upload Video</span>
              </button>
            </div>

            {/* Video Details Card */}
            {selectedVideoSource === 'sample' ? (
              <div className="rounded-xl border border-blue-100 bg-blue-50/60 p-3.5 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-blue-900">
                    Electronic City Traffic Sample
                  </span>
                  <span className="rounded-full bg-blue-200/80 px-2 py-0.5 text-[10px] font-bold text-blue-800">
                    Verified MP4
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] text-blue-950/80">
                  <div>
                    <span className="text-blue-700/80 block">Duration:</span>
                    <strong className="font-semibold">{currentVideo.duration}</strong>
                  </div>
                  <div>
                    <span className="text-blue-700/80 block">Frame Count:</span>
                    <strong className="font-semibold">{currentVideo.frames} frames</strong>
                  </div>
                  <div>
                    <span className="text-blue-700/80 block">Resolution:</span>
                    <strong className="font-semibold">{currentVideo.resolution}</strong>
                  </div>
                  <div>
                    <span className="text-blue-700/80 block">Framerate:</span>
                    <strong className="font-semibold">{currentVideo.fps}</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <input
                  type="file"
                  ref={fileInputRef}
                  accept=".mp4,.mov,.avi,.mkv,.webm"
                  onChange={handleFileSelected}
                  className="hidden"
                />

                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="cursor-pointer border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-4 text-center bg-slate-50/50 hover:bg-blue-50/30 transition-all"
                >
                  <div className="flex flex-col items-center gap-1.5">
                    <UploadCloud className="h-8 w-8 text-blue-600" />
                    <span className="text-xs font-semibold text-slate-700">
                      Click to choose video file
                    </span>
                    <span className="text-[10px] text-slate-500">
                      MP4, MOV, WebM (Max 50MB)
                    </span>
                  </div>
                </div>

                {isUploading && (
                  <div className="flex items-center gap-2 text-xs text-blue-600 font-medium">
                    <Loader2 className="h-4 w-4 animate-spin shrink-0" />
                    <span>Uploading and validating video…</span>
                  </div>
                )}

                {uploadError && (
                  <div className="flex items-start gap-1.5 text-xs text-rose-600 bg-rose-50 p-2.5 rounded-lg border border-rose-200">
                    <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{uploadError}</span>
                  </div>
                )}

                {uploadedVideo && (
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3 space-y-1.5 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-emerald-900 truncate">
                        {uploadedVideo.filename}
                      </span>
                      <span className="rounded-full bg-emerald-200 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                        Ready
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-[11px] text-emerald-800">
                      <span>{uploadedVideo.total_frames} frames</span>
                      <span>{uploadedVideo.duration_seconds}s</span>
                      <span>{uploadedVideo.width}x{uploadedVideo.height}</span>
                      <span>{uploadedVideo.fps} FPS</span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Section 2: Analysis Options */}
            <div className="border-t border-slate-100 pt-3 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                2. Evaluation Frame Span
              </h3>

              <div className="grid grid-cols-3 gap-1.5 p-1 bg-slate-100 rounded-xl text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => setFrameMode('30')}
                  className={`py-1.5 px-2 rounded-lg text-center transition-all ${
                    frameMode === '30'
                      ? 'bg-white text-blue-700 shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                  title="30 frames (~1.5s execution) ideal for fast interactive verification"
                >
                  <span className="block text-xs font-bold">30 Frames</span>
                  <span className="block text-[9px] text-slate-400 font-normal">
                    Quick Sample
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => setFrameMode('full')}
                  className={`py-1.5 px-2 rounded-lg text-center transition-all ${
                    frameMode === 'full'
                      ? 'bg-white text-blue-700 shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                  title="Processes all frames in video (~10-15s execution)"
                >
                  <span className="block text-xs font-bold">Full Video</span>
                  <span className="block text-[9px] text-slate-400 font-normal">
                    {currentVideo.frames} frames
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => setFrameMode('custom')}
                  className={`py-1.5 px-2 rounded-lg text-center transition-all ${
                    frameMode === 'custom'
                      ? 'bg-white text-blue-700 shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <span className="block text-xs font-bold">Custom</span>
                  <span className="block text-[9px] text-slate-400 font-normal">
                    Configurable
                  </span>
                </button>
              </div>

              {frameMode === 'custom' && (
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] font-semibold text-slate-600">
                    <span>Frame Count:</span>
                    <span className="font-bold text-blue-700">{customFrames} frames</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max={currentVideo.frames || 301}
                    value={customFrames}
                    onChange={(e) => setCustomFrames(Number(e.target.value))}
                    className="w-full accent-blue-600"
                  />
                </div>
              )}
            </div>

            {/* Advanced Hyperparameters Accordion */}
            <div className="border-t border-slate-100 pt-2 space-y-2">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center justify-between w-full text-xs font-semibold text-slate-600 hover:text-slate-900 py-1"
              >
                <span className="flex items-center gap-1.5">
                  <Sliders className="h-3.5 w-3.5 text-slate-400" />
                  <span>Pipeline Hyperparameters</span>
                </span>
                <span className="text-[11px] text-blue-600 font-medium">
                  {showAdvanced ? 'Hide' : 'Configure'}
                </span>
              </button>

              {showAdvanced && (
                <div className="space-y-3 pt-2 text-xs bg-slate-50 p-3 rounded-xl border border-slate-200/70">
                  {/* Confidence */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-600">
                      <span>Detection Confidence (conf):</span>
                      <span className="font-bold text-slate-800">{confThreshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={confThreshold}
                      onChange={(e) => setConfThreshold(Number(e.target.value))}
                      className="w-full accent-blue-600"
                    />
                  </div>

                  {/* IoU */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-600">
                      <span>NMS / Association IoU:</span>
                      <span className="font-bold text-slate-800">{iouThreshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.80"
                      step="0.05"
                      value={iouThreshold}
                      onChange={(e) => setIouThreshold(Number(e.target.value))}
                      className="w-full accent-blue-600"
                    />
                  </div>

                  {/* Min track lifespan */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-600">
                      <span>Min Track Lifespan Frames:</span>
                      <span className="font-bold text-slate-800">{minLifespan} frames</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="1"
                      value={minLifespan}
                      onChange={(e) => setMinLifespan(Number(e.target.value))}
                      className="w-full accent-blue-600"
                    />
                    <span className="text-[10px] text-slate-400 block leading-tight">
                      Tracks with observed frames &lt; threshold are pruned as noise.
                    </span>
                  </div>

                  {/* Annotated Video Checkbox */}
                  <label className="flex items-center gap-2 pt-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={generateAnnotated}
                      onChange={(e) => setGenerateAnnotated(e.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 accent-blue-600"
                    />
                    <span className="text-[11px] font-medium text-slate-700">
                      Render annotated video with bounding boxes & IDs
                    </span>
                  </label>
                </div>
              )}
            </div>

            {/* Error Message */}
            {analysisError && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-rose-600" />
                <div className="flex-1">
                  <span className="font-bold block">Analysis Failed</span>
                  <span>{analysisError}</span>
                </div>
              </div>
            )}

            {/* Main Action Button */}
            <button
              type="button"
              onClick={handleRunAnalysis}
              disabled={isAnalyzing || !status?.weights_available}
              className="flex w-full min-h-[46px] items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-bold text-white shadow-md hover:bg-blue-700 active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none transition-all cursor-pointer"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin shrink-0" />
                  <span>Analyzing Video…</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current shrink-0" />
                  <span>Run Computer Vision Analysis</span>
                </>
              )}
            </button>

            {/* Progress status notice while analyzing */}
            {isAnalyzing && (
              <div className="rounded-xl border border-blue-100 bg-blue-50/80 p-3 text-xs text-blue-800 space-y-1">
                <div className="flex items-center gap-2 font-semibold">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600 shrink-0" />
                  <span>{analysisProgressText || 'Processing frames with YOLO11n…'}</span>
                </div>
                <span className="text-[10px] text-blue-600 block">
                  Executing real inference on backend. UI remains fully responsive.
                </span>
              </div>
            )}
          </div>
        </aside>

        {/* Right Column: Player & Analysis Results */}
        <section className="space-y-5 min-w-0" aria-label="Computer Vision Video and Results">
          {/* Video Player */}
          <VisionVideoPlayer
            sourceVideoUrl={
              analysisResult
                ? analysisResult.source_video_url
                : currentVideo.previewUrl
            }
            annotatedVideoUrl={analysisResult?.annotated_video_url}
            metadata={
              analysisResult
                ? {
                    video_name: analysisResult.report.video_name,
                    video_fps: analysisResult.report.video_fps,
                    video_duration_seconds: analysisResult.report.video_duration_seconds,
                    total_frames_in_video: analysisResult.report.total_frames_in_video,
                    width: analysisResult.video_width || (selectedVideoSource === 'upload' && uploadedVideo ? uploadedVideo.width : sampleInfo?.width || 640),
                    height: analysisResult.video_height || (selectedVideoSource === 'upload' && uploadedVideo ? uploadedVideo.height : sampleInfo?.height || 360),
                  }
                : {
                    video_name: currentVideo.name,
                    video_fps: selectedVideoSource === 'upload' && uploadedVideo ? uploadedVideo.fps : sampleInfo?.fps || 29.97,
                    video_duration_seconds: selectedVideoSource === 'upload' && uploadedVideo ? uploadedVideo.duration_seconds : sampleInfo?.duration_seconds || 10.04,
                    total_frames_in_video: currentVideo.frames,
                    width: selectedVideoSource === 'upload' && uploadedVideo ? uploadedVideo.width : sampleInfo?.width || 640,
                    height: selectedVideoSource === 'upload' && uploadedVideo ? uploadedVideo.height : sampleInfo?.height || 360,
                  }
            }
            processedFrames={
              analysisResult?.report?.processed_frames ||
              (frameMode === '30' ? 30 : currentVideo.frames)
            }
          />

          {/* Results Display */}
          {analysisResult ? (
            <div className="space-y-5">
              {/* Statistical Metrics Highlights */}
              <VisionMetricsCards
                report={analysisResult.report}
                processingTime={analysisResult.processing_time_seconds}
              />

              {/* Per-Class Breakdown & Distribution */}
              <VisionClassBreakdown report={analysisResult.report} />

              {/* Detailed Tracklets Table */}
              <VisionTrackletTable
                tracklets={analysisResult.report.tracklets}
                reclassifiedDetections={analysisResult.report.reclassified_detections}
                minLifespan={analysisResult.report.min_track_lifespan_frames}
              />

              {/* About These Results & Disclaimers */}
              <VisionDisclaimer disclaimer={analysisResult.report.disclaimer} />
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white/70 p-8 text-center space-y-3">
              <div className="flex h-12 w-12 mx-auto items-center justify-center rounded-2xl bg-blue-50 text-blue-600 shadow-2xs">
                <Activity className="h-6 w-6" />
              </div>
              <div className="max-w-md mx-auto space-y-1">
                <h4 className="text-sm font-bold text-slate-800">
                  Ready for Real YOLO Inference
                </h4>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Select a frame count (e.g. Quick Sample 30 frames for fast testing or Full Video for complete analysis) and click{' '}
                  <strong>&quot;Run Computer Vision Analysis&quot;</strong> to execute the full vehicle detection and tracking pipeline.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
