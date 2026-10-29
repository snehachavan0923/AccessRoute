import { useEffect, useState } from 'react'
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ArrowRight,
  Database,
  Radio,
  ShieldCheck,
  RefreshCw,
  Info,
  Footprints,
} from 'lucide-react'
import { getObservationStatus } from '../../services/routes'

export default function VisionTrafficImpact({
  analysisResult,
  onNavigateToPlanner,
}) {
  const [liveStatus, setLiveStatus] = useState(null)
  const [loadingStatus, setLoadingStatus] = useState(false)

  function handleManualRefresh() {
    setLoadingStatus(true)
    getObservationStatus()
      .then((data) => {
        setLiveStatus(data)
        setLoadingStatus(false)
      })
      .catch((err) => {
        console.error('Failed to load observation status:', err)
        setLoadingStatus(false)
      })
  }

  useEffect(() => {
    let isMounted = true
    getObservationStatus()
      .then((data) => {
        if (isMounted) setLiveStatus(data)
      })
      .catch((err) => {
        console.error('Failed to load observation status:', err)
      })
    return () => {
      isMounted = false
    }
  }, [analysisResult])


  const publication = analysisResult?.traffic_publication
  const isPublished = publication?.published === true
  const hasAnalysis = Boolean(analysisResult)

  const isCalibrated = Boolean(
    publication?.published ||
    publication?.status === 'published' ||
    publication?.status === 'missing_congestion_estimate'
  )

  // Evaluation criteria for vision traffic eligibility
  const validationChecks = [
    {
      id: 'detection_tracking',
      label: 'Vehicle Detection & Tracking',
      passed: hasAnalysis && (analysisResult?.report?.tracklets?.length ?? 0) > 0,
      detail: hasAnalysis
        ? `${analysisResult?.report?.tracklets?.length ?? 0} vehicle tracks identified in analyzed frames using YOLO11n + ByteTrack.`
        : 'Awaiting video analysis run.',
    },
    {
      id: 'motion_analysis',
      label: 'Image-Plane Movement Analysis',
      passed: hasAnalysis && Boolean(analysisResult?.report?.motion_summary),
      detail: hasAnalysis && analysisResult?.report?.motion_summary
        ? `Movement analysis completed across ${analysisResult.report.motion_summary.total_evaluated_tracks} tracks (${analysisResult.report.motion_summary.moving_tracks_count} moving, ${analysisResult.report.motion_summary.stationary_tracks_count} stationary) in px and px/s.`
        : 'Awaiting movement analysis from tracking trajectories.',
    },
    {
      id: 'spatial_calib',
      label: 'Camera Spatial Calibration (Homography)',
      passed: isCalibrated,
      detail: isCalibrated
        ? 'Passed: Camera spatial calibration and road homography registered.'
        : 'Missing: No pixel-to-meter homography or camera geometry registered. Physical vehicle speeds (km/h) cannot be determined from 2D pixel displacement alone.',
      isWarning: !isCalibrated,
    },
    {
      id: 'osm_mapping',
      label: 'Directed Road-Segment Association',
      passed: Boolean(publication?.target_edge),
      detail: publication?.target_edge
        ? `Associated with directed edge [${publication.target_edge.join(', ')}].`
        : 'Missing: Video camera viewpoint is not registered to OpenStreetMap roadway coordinates.',
      isWarning: !publication?.target_edge,
    },
    {
      id: 'congestion_estimate',
      label: 'Validated Congestion Estimation',
      passed: publication?.congestion_index !== null && publication?.congestion_index !== undefined,
      detail:
        publication?.congestion_index !== null && publication?.congestion_index !== undefined
          ? `Verified congestion index: ${(publication.congestion_index * 100).toFixed(1)}%`
          : 'Missing: Real congestion estimation is not validated; bounding-box counts and pixel rates alone cannot establish roadway congestion.',
      isWarning: !(publication?.congestion_index !== null && publication?.congestion_index !== undefined),
    },
    {
      id: 'routing_eligibility',
      label: 'Vision Observation Routing Eligibility',
      passed: isPublished,
      detail: isPublished
        ? 'Eligible: Validated observation registered with VisionTrafficAdapter and active in routing engine.'
        : 'Ineligible: Route planner uses the simulated Electronic City dataset fallback. The shortest physical walking route remains unaffected.',
      isWarning: !isPublished,
    },
  ]

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-indigo-600 text-white shadow-2xs">
                <Radio className="h-3.5 w-3.5" />
              </span>
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-600">
                Area C — Traffic & Route Impact
              </span>
            </div>
            <h3 className="text-lg font-bold text-slate-900">
              Vision-to-Routing Validation & Safeguards
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed max-w-2xl">
              Inspect whether video observations satisfy the strict quality, calibration, and association
              requirements required before altering pedestrian route impedance.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleManualRefresh}
              disabled={loadingStatus}

              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-xs font-medium text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loadingStatus ? 'animate-spin' : ''}`} />
              <span>Refresh Status</span>
            </button>
          </div>
        </div>

        {/* Current Video Outcome Status Banner */}
        <div className="mt-4">
          {isPublished ? (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 p-4 text-emerald-950 flex items-start gap-3">
              <CheckCircle2 className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
              <div className="space-y-1">
                <span className="text-sm font-bold block">
                  Vision Observation Validated & Published
                </span>
                <p className="text-xs text-emerald-800 leading-relaxed">
                  This video passed spatial calibration and road association. It has published an active
                  traffic observation to directed edge [{publication?.target_edge?.join(', ')}] with
                  congestion index {publication?.congestion_index}.
                </p>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-slate-800 flex items-start gap-3">
              <ShieldCheck className="h-5 w-5 text-slate-500 mt-0.5 shrink-0" />
              <div className="space-y-1">
                <span className="text-sm font-bold block">
                  Safe Fallback Active: Simulated Electronic City Dataset Used
                </span>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {publication?.message ||
                    'Real video-derived routing is not yet enabled because camera calibration, road-segment association, and validated congestion estimation are missing. The route planner strictly uses the simulated Electronic City traffic dataset fallback, and the physical shortest walking route is never altered.'}
                </p>
                <div className="pt-1 flex flex-wrap items-center gap-2 text-[11px] font-semibold text-slate-500">
                  <span>Safeguard Status:</span>
                  <span className="font-mono bg-slate-200/80 px-2 py-0.5 rounded text-slate-700">
                    {publication?.status || (hasAnalysis ? 'uncalibrated_video_safeguard' : 'simulated_dataset_active')}
                  </span>
                  <span className="text-slate-400 font-normal">
                    (Dataset fallback active in route planner)
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Validation Gates Grid */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-4">
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Validation Safeguards Checklist
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {validationChecks.map((check) => (
            <div
              key={check.id}
              className={`rounded-xl border p-3.5 space-y-1.5 transition-all ${
                check.passed
                  ? 'border-emerald-200 bg-emerald-50/50'
                  : check.isWarning
                  ? 'border-amber-200 bg-amber-50/40'
                  : 'border-slate-200 bg-slate-50/50'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold text-slate-800">{check.label}</span>
                {check.passed ? (
                  <span className="flex items-center gap-1 text-[11px] font-bold text-emerald-700 bg-emerald-100/80 px-2 py-0.5 rounded-full">
                    <CheckCircle2 className="h-3 w-3" />
                    Passed
                  </span>
                ) : check.isWarning ? (
                  <span className="flex items-center gap-1 text-[11px] font-bold text-amber-800 bg-amber-100 px-2 py-0.5 rounded-full">
                    <AlertTriangle className="h-3 w-3" />
                    Rejected
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-600 bg-slate-200/80 px-2 py-0.5 rounded-full">
                    <XCircle className="h-3 w-3" />
                    Not Met
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">{check.detail}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Live Routing Engine Registry Status */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-blue-600" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Live Routing Engine Registry State
            </h4>
          </div>
          <span className="text-[11px] text-slate-500">
            Source: <code className="text-slate-700">/api/routes/observation-status</code>
          </span>
        </div>

        {/* Active Route Planner Traffic Source Indicator */}
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-1.5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Database className="h-4 w-4 text-blue-600" />
              Active Route Planner Traffic Data Source
            </span>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ${
                (liveStatus?.active_vision_observations_count ?? 0) > 0
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-blue-100 text-blue-800'
              }`}
            >
              {(liveStatus?.active_vision_observations_count ?? 0) > 0
                ? `Combined (${liveStatus.active_vision_observations_count} Vision Segments + Simulated Fallback)`
                : 'Simulated Dataset Fallback Active (Electronic City)'}
            </span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            {(liveStatus?.active_vision_observations_count ?? 0) > 0
              ? `The route planner is combining ${liveStatus.active_vision_observations_count} validated vision observation(s) on matching directed road segments with the simulated Electronic City traffic dataset fallback for all other corridors.`
              : 'The Electronic City simulated traffic dataset fallback is currently active for all route calculations. Vision-derived congestion is not yet available because uploaded/sample video footage lacks verified camera-to-road spatial calibration.'}
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Vision Traffic Adapter State */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-700">Vision Traffic Observations</span>
              <span className="rounded-full bg-blue-100 text-blue-800 px-2 py-0.5 text-xs font-bold">
                {liveStatus?.active_vision_observations_count ?? 0} active
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {(liveStatus?.active_vision_observations_count ?? 0) > 0
                ? `${liveStatus.active_vision_observations_count} active vision observation(s) currently influencing route costs.`
                : 'No active vision observations registered. Traffic-aware routes are using simulated dataset fallback.'}
            </p>
            {liveStatus?.active_vision_observations?.length > 0 && (
              <div className="space-y-1.5 pt-1">
                {liveStatus.active_vision_observations.map((obs, idx) => (
                  <div
                    key={idx}
                    className="text-[11px] font-mono bg-white border border-slate-200 p-2 rounded-lg flex items-center justify-between"
                  >
                    <span>Edge: ({obs.osm_u} ➜ {obs.osm_v})</span>
                    <span className="text-blue-700 font-bold">
                      Congestion: {(obs.experimental_congestion_index * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Pedestrian Accessibility Adapter State */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-700">Accessibility Obstacles</span>
              <span className="rounded-full bg-purple-100 text-purple-800 px-2 py-0.5 text-xs font-bold">
                {liveStatus?.active_accessibility_obstacles_count ?? 0} active
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {(liveStatus?.active_accessibility_obstacles_count ?? 0) > 0
                ? `${liveStatus.active_accessibility_obstacles_count} verified pedestrian barrier(s) influencing step-free routes.`
                : 'No verified pedestrian pathway obstacles currently registered.'}
            </p>
            {liveStatus?.active_accessibility_obstacles?.length > 0 && (
              <div className="space-y-1.5 pt-1">
                {liveStatus.active_accessibility_obstacles.map((obs, idx) => (
                  <div
                    key={idx}
                    className="text-[11px] font-mono bg-white border border-slate-200 p-2 rounded-lg flex items-center justify-between"
                  >
                    <span>Edge: ({obs.osm_u} ➜ {obs.osm_v})</span>
                    <span className="text-purple-700 font-bold capitalize">
                      {obs.obstacle_type.replace(/_/g, ' ')} ({obs.severity})
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Link to Route Planner */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-slate-100">
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <Info className="h-4 w-4 text-slate-400 shrink-0" />
            <span>
              Route comparisons show full data-source attribution and explain detour choices.
            </span>
          </div>

          {onNavigateToPlanner && (
            <button
              type="button"
              onClick={onNavigateToPlanner}
              className="flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-blue-600 text-white text-xs font-bold shadow-sm hover:bg-blue-700 transition-colors cursor-pointer shrink-0"
            >
              <Footprints className="h-4 w-4" />
              <span>Open Route Planner</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
