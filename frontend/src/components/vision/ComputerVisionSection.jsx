import { useEffect, useRef, useState } from 'react'
import {
  Sparkles,
  Play,
  UploadCloud,
  FileVideo,
  Loader2,
  AlertCircle,
  Activity,
  Video,
  Radio,
  SlidersHorizontal,
} from 'lucide-react'

import {
  analyzeVideo,
  getVisionStatus,
  uploadVideo,
} from '../../services/vision'
import VisionVideoPlayer from './VisionVideoPlayer'
import VisionMetricsCards from './VisionMetricsCards'
import VisionMotionSummary from './VisionMotionSummary'
import VisionClassBreakdown from './VisionClassBreakdown'
import VisionTrackletTable from './VisionTrackletTable'
import VisionDisclaimer from './VisionDisclaimer'
import VisionTrafficImpact from './VisionTrafficImpact'
import { ENABLE_TRAFFIC_IMPACT_TAB } from '../../config/features'

export default function ComputerVisionSection({
  onNavigateToPlanner,
  showTrafficImpactTab = ENABLE_TRAFFIC_IMPACT_TAB,
}) {
  const [status, setStatus] = useState(null)
  const [statusLoading, setStatusLoading] = useState(true)

  // Subsystem Tabs: 'analysis' (A), 'motion' (B), optional 'impact' (C, feature-flagged)
  const [activeArea, setActiveArea] = useState('analysis')
  const currentArea = !showTrafficImpactTab && activeArea === 'impact' ? 'analysis' : activeArea

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
        setAnalysisResult(null) // Clears stale results on video change
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
    setAnalysisResult(null)
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
      publishToTrafficRouter: false, // Default: uncalibrated uploads do not alter routes
    }

    const progressTimer = setTimeout(() => {
      setAnalysisProgressText('Tracking vehicle trajectories with ByteTrack…')
    }, 1500)

    const postProcTimer = setTimeout(() => {
      setAnalysisProgressText(
        'Post-processing tracklets & analyzing motion displacement…',
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
            Real-Time Vehicle Detection & Motion Diagnostics
          </h2>

          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Run real neural-network inference using local YOLO11n weights and ByteTrack multi-object tracking.
            Evaluate raw detection events, trajectory lifespans, and image-plane motion displacement on traffic footage.
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

      {/* Logical Area Selector Tabs: Area A, Area B, Area C */}
      <div className="border-b border-slate-200">
        <nav className="flex space-x-2 sm:space-x-4" aria-label="Computer Vision Subsystems">
          <button
            type="button"
            onClick={() => setActiveArea('analysis')}
            className={`flex items-center gap-2 pb-3 px-3 text-xs sm:text-sm font-bold border-b-2 transition-all cursor-pointer ${
              currentArea === 'analysis'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300'
            }`}
          >
            <Video className="h-4 w-4" />
            <span>A. Video Analysis</span>
            {analysisResult && (
              <span className="rounded-full bg-blue-100 text-blue-700 px-1.5 py-0.2 text-[10px]">
                {analysisResult.report.total_vehicle_detections} dets
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={() => setActiveArea('motion')}
            className={`flex items-center gap-2 pb-3 px-3 text-xs sm:text-sm font-bold border-b-2 transition-all cursor-pointer ${
              currentArea === 'motion'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300'
            }`}
          >
            <Activity className="h-4 w-4" />
            <span>B. Motion Diagnostics</span>
            {analysisResult?.report?.motion_summary && (
              <span className="rounded-full bg-emerald-100 text-emerald-800 px-1.5 py-0.2 text-[10px]">
                {analysisResult.report.motion_summary.total_evaluated_tracks} tracks
              </span>
            )}
          </button>

          {showTrafficImpactTab && (
            <button
              type="button"
              onClick={() => setActiveArea('impact')}
              className={`flex items-center gap-2 pb-3 px-3 text-xs sm:text-sm font-bold border-b-2 transition-all cursor-pointer ${
                currentArea === 'impact'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300'
              }`}
            >
              <Radio className="h-4 w-4" />
              <span>C. Traffic & Route Impact</span>
              <span className="rounded-full bg-slate-100 text-slate-600 px-1.5 py-0.2 text-[10px]">
                Safeguards
              </span>
            </button>
          )}
        </nav>
      </div>

      {/* AREA A: Video Analysis */}
      {currentArea === 'analysis' && (
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
                    if (selectedVideoSource !== 'sample') {
                      setSelectedVideoSource('sample')
                      setAnalysisResult(null)
                      setAnalysisError('')
                    }
                  }}
                  className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition-all cursor-pointer ${
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
                    if (selectedVideoSource !== 'upload') {
                      setSelectedVideoSource('upload')
                      setAnalysisResult(null)
                      setAnalysisError('')
                    }
                  }}
                  className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition-all cursor-pointer ${
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
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-blue-900">{currentVideo.name}</span>
                    <span className="rounded bg-blue-200/60 px-1.5 py-0.5 text-[10px] font-bold text-blue-800">
                      Sample Asset
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] text-blue-900/80">
                    <div>Duration: <strong>{currentVideo.duration}</strong></div>
                    <div>Frames: <strong>{currentVideo.frames}</strong></div>
                    <div>Resolution: <strong>{currentVideo.resolution}</strong></div>
                    <div>Frame Rate: <strong>{currentVideo.fps}</strong></div>
                  </div>
                </div>
              ) : (
                <div className="space-y-2.5">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-slate-300 hover:border-blue-400 bg-slate-50 hover:bg-blue-50/30 rounded-xl p-4 text-center cursor-pointer transition-colors space-y-1.5"
                  >
                    <UploadCloud className="h-6 w-6 text-slate-400 mx-auto" />
                    <span className="text-xs font-semibold text-slate-700 block">
                      {isUploading ? 'Uploading Video…' : 'Click to Upload MP4 / MOV Video'}
                    </span>
                    <span className="text-[10px] text-slate-400 block">
                      Max file size: 50 MB
                    </span>
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-matroska,video/webm"
                    onChange={handleFileSelected}
                    className="hidden"
                  />

                  {uploadError && (
                    <div className="rounded-lg border border-rose-200 bg-rose-50 p-2 text-xs text-rose-700 flex items-start gap-1.5">
                      <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-rose-600" />
                      <span>{uploadError}</span>
                    </div>
                  )}

                  {uploadedVideo && (
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3 text-xs text-emerald-950 space-y-1">
                      <span className="font-bold block truncate">{uploadedVideo.filename}</span>
                      <div className="grid grid-cols-2 gap-1 text-[11px] text-emerald-900">
                        <div>Duration: <strong>{uploadedVideo.duration_seconds}s</strong></div>
                        <div>Frames: <strong>{uploadedVideo.total_frames}</strong></div>
                        <div>Resolution: <strong>{uploadedVideo.width}x{uploadedVideo.height}</strong></div>
                        <div>FPS: <strong>{uploadedVideo.fps}</strong></div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 2. Select Evaluation Frame-Span */}
              <div className="border-t border-slate-100 pt-3 space-y-2.5">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  2. Evaluation Frame-Span
                </h3>

                <div className="grid grid-cols-3 gap-1.5 text-xs font-semibold">
                  <button
                    type="button"
                    onClick={() => setFrameMode('30')}
                    className={`py-2 px-2 text-center rounded-lg border transition-all cursor-pointer ${
                      frameMode === '30'
                        ? 'border-blue-600 bg-blue-50 text-blue-700 font-bold shadow-2xs'
                        : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    Quick (30f)
                  </button>

                  <button
                    type="button"
                    onClick={() => setFrameMode('full')}
                    className={`py-2 px-2 text-center rounded-lg border transition-all cursor-pointer ${
                      frameMode === 'full'
                        ? 'border-blue-600 bg-blue-50 text-blue-700 font-bold shadow-2xs'
                        : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    Full Video
                  </button>

                  <button
                    type="button"
                    onClick={() => setFrameMode('custom')}
                    className={`py-2 px-2 text-center rounded-lg border transition-all cursor-pointer ${
                      frameMode === 'custom'
                        ? 'border-blue-600 bg-blue-50 text-blue-700 font-bold shadow-2xs'
                        : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    Custom
                  </button>
                </div>

                {frameMode === 'custom' && (
                  <div className="flex items-center gap-2 pt-1">
                    <label htmlFor="custom-frames-input" className="text-xs text-slate-600 whitespace-nowrap">
                      Max Frames:
                    </label>
                    <input
                      id="custom-frames-input"
                      type="number"
                      min={1}
                      max={currentVideo.frames || 600}
                      value={customFrames}
                      onChange={(e) => setCustomFrames(Number(e.target.value))}
                      className="w-full rounded-lg border border-slate-200 py-1 px-2 text-xs font-semibold text-slate-800"
                    />
                  </div>
                )}
              </div>

              {/* Advanced Parameters Toggle */}
              <div className="border-t border-slate-100 pt-3">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 cursor-pointer"
                >
                  <SlidersHorizontal className="h-3.5 w-3.5" />
                  <span>{showAdvanced ? 'Hide Advanced Settings' : 'Advanced Detection Settings'}</span>
                </button>

                {showAdvanced && (
                  <div className="pt-3 space-y-3">
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px] text-slate-600">
                        <span>Confidence Threshold:</span>
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

                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px] text-slate-600">
                        <span>NMS IoU Threshold:</span>
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

                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px] text-slate-600">
                        <span>Min Track Lifespan:</span>
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
                    </div>

                    <label className="flex items-center gap-2 pt-1 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={generateAnnotated}
                        onChange={(e) => setGenerateAnnotated(e.target.checked)}
                        className="h-4 w-4 rounded border-slate-300 text-blue-600 accent-blue-600"
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

              {/* Run Action Button */}
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

          {/* Right Column: Player & Compact Detection Summary */}
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
                      width: analysisResult.video_width || 640,
                      height: analysisResult.video_height || 360,
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

            {/* Compact Detection & Tracking Summary */}
            {analysisResult ? (
              <div className="space-y-4">
                <VisionMetricsCards
                  report={analysisResult.report}
                  processingTime={analysisResult.processing_time_seconds}
                />
                <VisionClassBreakdown report={analysisResult.report} />
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
                    Select a frame count (e.g. Quick 30 frames for fast sample or Full Video) and click{' '}
                    <strong>&quot;Run Computer Vision Analysis&quot;</strong> to evaluate vehicle detection and multi-object tracking.
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>
      )}

      {/* AREA B: Motion Diagnostics */}
      {currentArea === 'motion' && (
        <div className="space-y-5">
          {analysisResult ? (
            <>
              {/* Scene-Level Motion Diagnostics */}
              <VisionMotionSummary
                motionSummary={analysisResult.report?.motion_summary}
              />

              {/* Per-Track Displacement & Details */}
              <VisionTrackletTable
                tracklets={analysisResult.report.tracklets}
                reclassifiedDetections={analysisResult.report.reclassified_detections}
                minLifespan={analysisResult.report.min_track_lifespan_frames}
              />
            </>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white/70 p-12 text-center space-y-3">
              <div className="flex h-12 w-12 mx-auto items-center justify-center rounded-2xl bg-slate-100 text-slate-500 shadow-2xs">
                <Activity className="h-6 w-6" />
              </div>
              <div className="max-w-md mx-auto space-y-1">
                <h4 className="text-sm font-bold text-slate-800">
                  No Motion Data Available Yet
                </h4>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Switch to <strong>Area A (Video Analysis)</strong> and run computer vision analysis to compute
                  per-vehicle pixel displacement, movement rates (px/s), and stationary classifications.
                </p>
                <div className="pt-2">
                  <button
                    type="button"
                    onClick={() => setActiveArea('analysis')}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 transition-colors cursor-pointer"
                  >
                    <Video className="h-3.5 w-3.5" />
                    <span>Go to Video Analysis</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* AREA C: Traffic & Route Impact (Temporarily hidden from presentation UI by default) */}
      {showTrafficImpactTab && currentArea === 'impact' && (
        <VisionTrafficImpact
          analysisResult={analysisResult}
          onNavigateToPlanner={onNavigateToPlanner}
        />
      )}
    </div>
  )
}
