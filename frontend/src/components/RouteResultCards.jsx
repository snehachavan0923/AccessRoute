import { useState } from 'react'
import {
  Info,
  ChevronDown,
  ChevronRight,
  Check,
  AlertCircle,
  ShieldAlert,
  Footprints,
  Radio,
  Database,
  Sparkles,
} from 'lucide-react'

function formatDistance(meters) {
  if (meters === undefined || meters === null) return '—'
  return meters >= 1000
    ? `${(meters / 1000).toFixed(2)} km`
    : `${Math.round(meters)} m`
}

function getSourceBadge(source) {
  switch (source) {
    case 'vision':
      return {
        label: 'Vision-Derived Traffic',
        classes: 'bg-indigo-100 text-indigo-800 border-indigo-200',
        icon: Radio,
        description: 'Validated computer vision observation active on road segment(s)',
      }
    case 'mixed':
      return {
        label: 'Mixed: Vision + Simulated',
        classes: 'bg-purple-100 text-purple-800 border-purple-200',
        icon: Sparkles,
        description: 'Combines vision observations with simulated dataset fallback',
      }
    case 'simulated':
      return {
        label: 'Simulated Dataset Fallback',
        classes: 'bg-slate-100 text-slate-700 border-slate-200',
        icon: Database,
        description: 'Electronic City synthetic traffic model (camera data uncalibrated or unavailable)',
      }
    default:
      return {
        label: 'No Traffic Data',
        classes: 'bg-slate-100 text-slate-600 border-slate-200',
        icon: Info,
        description: 'Traffic unknown for traversed corridor; physical distance applied',
      }
  }
}

export default function RouteResultCards({ comparison, route }) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false)

  // Single shortest route calculated only
  if (!comparison && route) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-3.5 sm:p-4 shadow-xs space-y-2 max-w-full min-w-0">
        <div className="flex flex-wrap items-center justify-between gap-2 min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="h-3 w-3 rounded-full bg-blue-600 ring-2 ring-blue-100 shrink-0" />
            <h3 className="text-sm font-bold text-slate-800 truncate">Shortest Walking Route Baseline</h3>
          </div>
          <span className="text-base font-bold text-blue-600 shrink-0">
            {formatDistance(route.distance_meters)}
          </span>
        </div>
        <p className="text-xs text-slate-500 leading-relaxed">
          The shortest walking distance calculated on the OpenStreetMap pedestrian network.
        </p>
      </div>
    )
  }

  if (!comparison) return null

  const shortest = comparison.shortest_distance
  const trafficAware = comparison.traffic_accessibility_aware
  const isAlternativeFound = comparison.status === 'alternative_found'
  const isIdentical = comparison.status === 'identical_routes'
  const isOutsideCoverage = comparison.status === 'no_traffic_coverage_in_area'
  const isOffline =
    comparison.status === 'traffic_dataset_unavailable' ||
    comparison.status === 'partial_shortest_only'

  const shortestDist = shortest?.distance_meters
  const trafficDist = trafficAware?.distance_meters
  const distanceDelta =
    shortestDist !== undefined && trafficDist !== undefined
      ? trafficDist - shortestDist
      : null

  const sourceBadge = getSourceBadge(comparison.primary_traffic_source || trafficAware?.traffic_source)
  const SourceIcon = sourceBadge.icon

  const isStepFree = comparison.accessibility_mode === 'step_free'

  return (
    <div className="space-y-3.5 max-w-full min-w-0">
      {/* Primary Explainability Banner */}
      <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-3.5 text-xs text-blue-950 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-bold text-xs uppercase tracking-wider text-blue-800 flex items-center gap-1.5">
            <Info className="h-3.5 w-3.5 text-blue-600" />
            Route Recommendation Rationale
          </span>

          <div className="flex flex-wrap items-center gap-1.5">
            {/* Traffic Data Source Badge */}
            <span
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${sourceBadge.classes}`}
              title={sourceBadge.description}
            >
              <SourceIcon className="h-3 w-3" />
              {sourceBadge.label}
            </span>

            {/* Accessibility Mode Badge */}
            {isStepFree && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-purple-100 text-purple-800 border border-purple-200">
                <Footprints className="h-3 w-3" />
                Step-Free Mode
              </span>
            )}
          </div>
        </div>

        <p className="text-xs text-blue-900 leading-relaxed font-medium">
          {comparison.explanation ||
            (isAlternativeFound
              ? `An alternative pedestrian route was selected (+${formatDistance(distanceDelta)} detour) to minimize travel impedance.`
              : 'The shortest pedestrian path is also recommended under current traffic and accessibility conditions.')}
        </p>

        {/* Explainability footnote distinguishing simulated vs measured */}
        <div className="pt-1 border-t border-blue-100/80 text-[11px] text-blue-700/90 leading-tight">
          {comparison.primary_traffic_source === 'vision' ? (
            <span>
              Validated computer vision observation applied to monitored segment(s). Uncalibrated segments safely use simulated fallback.
            </span>
          ) : (
            <span>
              Traffic impedance is based on simulated dataset fallback. Vision observations are currently unavailable or rejected by safety calibration gates.
            </span>
          )}
        </div>
      </div>

      {/* Dynamic Status / Context Notice */}
      {isOutsideCoverage && (
        <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50/70 p-3 text-xs text-amber-900 max-w-full min-w-0">
          <Info className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
          <div className="min-w-0 break-words flex-1">
            <strong>Pedestrian route available; traffic data unavailable for this area.</strong>
            <p className="text-[11px] text-amber-700 mt-0.5 leading-relaxed">
              Traffic estimates cover roads within Electronic City only. Using the shortest physical walking route.
            </p>
          </div>
        </div>
      )}

      {isIdentical && (
        <div className="flex items-start gap-2 rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700 max-w-full min-w-0">
          <Check className="h-4 w-4 text-emerald-600 mt-0.5 shrink-0" />
          <div className="min-w-0 break-words flex-1">
            <strong>Shortest walking route is optimal under evaluated conditions.</strong>
            <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">
              No significant traffic impedance or accessibility barriers were encountered on the shortest route.
            </p>
          </div>
        </div>
      )}

      {isOffline && (
        <div className="flex items-start gap-2 rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700 max-w-full min-w-0">
          <AlertCircle className="h-4 w-4 text-slate-500 mt-0.5 shrink-0" />
          <div className="min-w-0 break-words flex-1">
            <strong>Traffic alternative unavailable (showing shortest walking route fallback).</strong>
          </div>
        </div>
      )}

      {/* Side-by-Side (desktop) or Stacked (mobile) Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-full min-w-0">
        {/* Blue Card: Shortest Walking Route Baseline */}
        <div className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-3.5 sm:p-4 shadow-xs transition-shadow hover:shadow-sm max-w-full min-w-0">
          <div className="min-w-0">
            {/* Header: Title and badge */}
            <div className="flex items-start justify-between gap-2 pb-2 border-b border-slate-100 min-w-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="h-3 w-3 rounded-full bg-blue-600 ring-2 ring-blue-100 shrink-0" />
                <h4 className="text-sm font-bold text-slate-900 leading-snug break-words">
                  Shortest Walking Baseline
                </h4>
              </div>
              <span className="rounded bg-blue-50 px-2 py-0.5 text-[10px] font-semibold text-blue-700 shrink-0">
                Baseline
              </span>
            </div>

            <p className="mt-2 text-xs text-slate-500 leading-relaxed">
              Physical distance minimization. Does not avoid road traffic or staircase barriers.
            </p>

            {/* Metrics */}
            <div className="mt-3 space-y-2 max-w-full min-w-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5 min-w-0">
                <span className="text-xs text-slate-500">Physical distance</span>
                <span className="text-lg font-bold text-blue-600 shrink-0">
                  {formatDistance(shortest?.distance_meters)}
                </span>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Detour vs baseline</span>
                <span className="font-medium text-slate-600 shrink-0">0 m (Baseline)</span>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Road segments</span>
                <span className="font-medium text-slate-800 shrink-0">
                  {shortest?.edges_count} segments
                </span>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Traffic-monitored segments</span>
                <span className="font-medium text-slate-800 shrink-0">
                  {shortest?.matched_traffic_edges_count} of {shortest?.edges_count}
                  {shortest?.vision_matched_edges_count > 0 && (
                    <span className="ml-1 text-[10px] text-indigo-600 font-bold">
                      ({shortest.vision_matched_edges_count} vision)
                    </span>
                  )}
                </span>
              </div>

              {/* Staircase & obstacle segment indicators */}
              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Stairs / Barriers</span>
                <span className={`font-medium shrink-0 ${shortest?.steps_count > 0 ? 'text-amber-700 font-semibold' : 'text-slate-800'}`}>
                  {shortest?.steps_count ?? 0} stairs, {shortest?.obstacles_count ?? 0} obstacles
                </span>
              </div>

              <div className="rounded-lg bg-slate-50 p-2 text-xs max-w-full min-w-0">
                <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 min-w-0">
                  <span className="text-slate-600">
                    {comparison?.primary_traffic_source === 'simulated'
                      ? 'Average congestion (Simulated Dataset)'
                      : comparison?.primary_traffic_source === 'vision'
                      ? 'Average congestion (Vision Traffic)'
                      : 'Average monitored congestion'}
                  </span>
                  <span className="font-semibold text-slate-900 shrink-0">
                    {shortest?.average_congestion !== null
                      ? `${(shortest.average_congestion * 100).toFixed(1)}%`
                      : 'Unknown'}
                  </span>
                </div>
                <p className="mt-0.5 text-[10px] text-slate-400 break-words leading-tight">
                  {shortest?.average_congestion !== null
                    ? comparison?.primary_traffic_source === 'simulated'
                      ? `Simulated estimate from Electronic City dataset across ${shortest.matched_traffic_edges_count} covered segments (not live camera)`
                      : `Estimated across ${shortest.matched_traffic_edges_count} covered segments`
                    : 'Unmonitored segments have unknown traffic; not zero'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Green Card: Traffic & Accessibility Aware Recommendation */}
        <div className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-3.5 sm:p-4 shadow-xs transition-shadow hover:shadow-sm max-w-full min-w-0">
          <div className="min-w-0">
            {/* Header: Title and badge */}
            <div className="flex items-start justify-between gap-2 pb-2 border-b border-slate-100 min-w-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="h-3 w-3 rounded-full bg-emerald-600 ring-2 ring-emerald-100 shrink-0" />
                <h4 className="text-sm font-bold text-slate-900 leading-snug break-words">
                  Recommended Route
                </h4>
              </div>
              <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 shrink-0">
                {isAlternativeFound ? 'Optimized Detour' : 'Recommended'}
              </span>
            </div>

            <p className="mt-2 text-xs text-slate-500 leading-relaxed">
              {isStepFree
                ? 'Accounts for vehicular traffic and penalizes stairs/obstacles for step-free mobility.'
                : 'Balanced pedestrian routing avoiding heavier vehicular congestion where data is available.'}
            </p>

            {/* Metrics */}
            <div className="mt-3 space-y-2 max-w-full min-w-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5 min-w-0">
                <span className="text-xs text-slate-500">Physical distance</span>
                <span className="text-lg font-bold text-emerald-600 shrink-0">
                  {formatDistance(trafficAware?.distance_meters)}
                </span>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Additional distance</span>
                <span className="font-medium text-slate-800 shrink-0">
                  {distanceDelta !== null && distanceDelta > 0 ? (
                    <span className="text-amber-700 font-semibold">
                      +{formatDistance(distanceDelta)}{' '}
                      {shortestDist ? `(+${((distanceDelta / shortestDist) * 100).toFixed(1)}%)` : ''}
                    </span>
                  ) : distanceDelta === 0 || isIdentical ? (
                    <span className="text-slate-500">Identical (0 m)</span>
                  ) : (
                    '—'
                  )}
                </span>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Road segments</span>
                <span className="font-medium text-slate-800 shrink-0">
                  {trafficAware?.edges_count} segments
                </span>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Traffic-monitored segments</span>
                <span className="font-medium text-slate-800 shrink-0">
                  {trafficAware?.matched_traffic_edges_count} of {trafficAware?.edges_count}
                  {trafficAware?.vision_matched_edges_count > 0 && (
                    <span className="ml-1 text-[10px] text-indigo-600 font-bold">
                      ({trafficAware.vision_matched_edges_count} vision)
                    </span>
                  )}
                </span>
              </div>

              {/* Staircase & obstacle segment indicators */}
              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 text-xs min-w-0">
                <span className="text-slate-500">Stairs / Barriers</span>
                <span className={`font-medium shrink-0 ${trafficAware?.steps_count === 0 && isStepFree ? 'text-emerald-700 font-bold' : 'text-slate-800'}`}>
                  {trafficAware?.steps_count ?? 0} stairs, {trafficAware?.obstacles_count ?? 0} obstacles
                </span>
              </div>

              <div className="rounded-lg bg-emerald-50/50 border border-emerald-100 p-2 text-xs max-w-full min-w-0">
                <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5 min-w-0">
                  <span className="text-emerald-900 font-medium">
                    {comparison?.primary_traffic_source === 'simulated' || trafficAware?.traffic_source === 'simulated'
                      ? 'Average congestion (Simulated Dataset)'
                      : comparison?.primary_traffic_source === 'vision' || trafficAware?.traffic_source === 'vision'
                      ? 'Average congestion (Vision Traffic)'
                      : 'Average monitored congestion'}
                  </span>
                  <span className="font-bold text-emerald-700 shrink-0">
                    {trafficAware?.average_congestion !== null
                      ? `${(trafficAware.average_congestion * 100).toFixed(1)}%`
                      : 'Unknown'}
                  </span>
                </div>
                <p className="mt-0.5 text-[10px] text-emerald-700/70 break-words leading-tight">
                  {trafficAware?.average_congestion !== null
                    ? (comparison?.primary_traffic_source === 'simulated' || trafficAware?.traffic_source === 'simulated')
                      ? `Simulated estimate from Electronic City dataset across ${trafficAware.matched_traffic_edges_count} covered segments (not live camera)`
                      : `Estimated across ${trafficAware.matched_traffic_edges_count} covered segments`
                    : 'Unmonitored segments have unknown traffic; not zero'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Accessibility Disclaimer Notice */}
      <div className="flex items-start gap-2 rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600 max-w-full min-w-0">
        <ShieldAlert className="h-4 w-4 text-slate-400 mt-0.5 shrink-0" />
        <p className="leading-relaxed">
          <strong className="text-slate-700">Accessibility & Traffic Data Scope:</strong>{' '}
          Accessibility coverage is incomplete; missing obstacle information does not mean a route is accessible.
          Vehicular traffic data uses the simulated Electronic City dataset fallback unless a calibrated vision observation is active.
        </p>
      </div>

      {/* Technical Cost Breakdown Accordion */}
      <div className="pt-0.5 max-w-full min-w-0">
        <button
          type="button"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700 transition-colors py-1 cursor-pointer"
        >
          {showTechnicalDetails ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0" />
          )}
          <span>Technical Cost Breakdown & Graph Attributes</span>
        </button>

        {showTechnicalDetails && (
          <div className="mt-1.5 rounded-xl border border-slate-200 bg-slate-50 p-3.5 text-xs text-slate-600 space-y-2.5 max-w-full min-w-0 overflow-x-auto">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pb-2 border-b border-slate-200">
              <div className="space-y-1">
                <span className="font-bold text-slate-800 block">Shortest Route Cost Breakdown:</span>
                <div>Distance: {shortest?.distance_meters?.toFixed(1)} m</div>
                <div>Traffic Penalty: {shortest?.traffic_cost_penalty?.toFixed(1)}</div>
                <div>Accessibility Penalty: {shortest?.accessibility_cost_penalty?.toFixed(1)}</div>
                <div className="font-semibold text-slate-900">Total Cost: {shortest?.cost?.toFixed(1)}</div>
              </div>

              <div className="space-y-1">
                <span className="font-bold text-slate-800 block">Recommended Route Cost Breakdown:</span>
                <div>Distance: {trafficAware?.distance_meters?.toFixed(1)} m</div>
                <div>Traffic Penalty: {trafficAware?.traffic_cost_penalty?.toFixed(1)}</div>
                <div>Accessibility Penalty: {trafficAware?.accessibility_cost_penalty?.toFixed(1)}</div>
                <div className="font-semibold text-slate-900">Total Cost: {trafficAware?.cost?.toFixed(1)}</div>
              </div>
            </div>

            <div className="text-[11px] space-y-1">
              <div>
                <strong>Primary Traffic Source:</strong>{' '}
                <span className="font-mono">{comparison.primary_traffic_source}</span>
              </div>
              <div>
                <strong>Vision Adapter Active Observation:</strong>{' '}
                <span>{comparison.vision_observation_active ? 'Yes' : 'No'}</span>
              </div>
              <div>
                <strong>Applied Constraints:</strong>{' '}
                <span>
                  {trafficAware?.accessibility_constraints_applied?.length > 0
                    ? trafficAware.accessibility_constraints_applied.join(', ')
                    : 'None (Standard Mode)'}
                </span>
              </div>
              <div>
                <strong>Cost Formula:</strong>{' '}
                <span className="font-mono block text-slate-700 pt-0.5">
                  {comparison.cost_formula}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
