import { useEffect, useRef, useState } from 'react'
import LocationInput from './components/LocationInput'
import MapView from './components/map/MapView'
import { ELECTRONIC_CITY_DEFAULT_BOUNDS } from './components/map/mapConfig'
import {
  calculateShortestRoute,
  compareRoutes,
  getTrafficInfo,
} from './services/routes'

// Sample preset routes within Electronic City for instant 1-click testing
const PRESET_ROUTES = [
  {
    id: 'toll_to_velankani',
    label: 'Toll Gate ➜ Velankani',
    origin: {
      latitude: 12.847657,
      longitude: 77.671263,
      label: 'Electronic City Toll Gate',
    },
    destination: {
      latitude: 12.8398,
      longitude: 77.6775,
      label: 'Velankani Tech Park',
    },
  },
  {
    id: 'infy_to_wipro',
    label: 'Infosys ➜ Wipro',
    origin: {
      latitude: 12.83852,
      longitude: 77.6658,
      label: 'Infosys Gate 1',
    },
    destination: {
      latitude: 12.8421,
      longitude: 77.675,
      label: 'Wipro Technologies Campus',
    },
  },
  {
    id: 'toll_to_doddathoguru',
    label: 'Toll Gate ➜ Doddathoguru',
    origin: {
      latitude: 12.847657,
      longitude: 77.671263,
      label: 'Electronic City Toll Gate',
    },
    destination: {
      latitude: 12.848,
      longitude: 77.683,
      label: 'Doddathoguru Junction',
    },
  },
]

function formatDistance(meters) {
  if (meters === undefined || meters === null) return '—'
  return meters >= 1_000
    ? `${(meters / 1_000).toFixed(2)} km`
    : `${meters.toFixed(0)} m`
}

export default function App() {
  const [origin, setOrigin] = useState(null)
  const [destination, setDestination] = useState(null)
  const [selectionTarget, setSelectionTarget] = useState('origin')
  const [route, setRoute] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [isCalculating, setIsCalculating] = useState(false)
  const [calculatingMode, setCalculatingMode] = useState('')
  const [routeError, setRouteError] = useState('')
  const [manualCameraAction, setManualCameraAction] = useState(null)

  // Traffic settings
  const [trafficInfo, setTrafficInfo] = useState(null)
  const [selectedTimestamp, setSelectedTimestamp] = useState('2026-01-01 08:30:00')
  const [trafficWeight, setTrafficWeight] = useState(2.0)

  const requestController = useRef(null)

  // Fetch dataset info on initial mount
  useEffect(() => {
    const controller = new AbortController()
    getTrafficInfo(controller.signal)
      .then((info) => {
        setTrafficInfo(info)
        if (info?.sample_presets?.length > 0) {
          setSelectedTimestamp(info.sample_presets[0].timestamp)
        }
      })
      .catch(() => {
        // Safe fallback if traffic info endpoint is unreachable
      })

    return () => controller.abort()
  }, [])

  function clearRouteRequest() {
    requestController.current?.abort()
    requestController.current = null
    setIsCalculating(false)
    setCalculatingMode('')
  }

  function handleLocationSelectFromMap(location) {
    clearRouteRequest()
    setRoute(null)
    setComparison(null)
    setRouteError('')
    setManualCameraAction(null)

    const formattedLocation = {
      ...location,
      label: `Map Point (${location.latitude.toFixed(5)}, ${location.longitude.toFixed(5)})`,
    }

    if (selectionTarget === 'origin') {
      setOrigin(formattedLocation)
      setSelectionTarget('destination') // Seamless auto-advance
    } else {
      setDestination(formattedLocation)
    }
  }

  function handleReset() {
    clearRouteRequest()
    setOrigin(null)
    setDestination(null)
    setRoute(null)
    setComparison(null)
    setRouteError('')
    setSelectionTarget('origin')
    setManualCameraAction(null)
  }

  function handleApplyPreset(preset) {
    clearRouteRequest()
    setOrigin(preset.origin)
    setDestination(preset.destination)
    setRoute(null)
    setComparison(null)
    setRouteError('')
    setManualCameraAction(null)
  }

  function handleGoToElectronicCity() {
    const bounds = trafficInfo?.bounds
      ? [
          [trafficInfo.bounds.min_lat, trafficInfo.bounds.min_lon],
          [trafficInfo.bounds.max_lat, trafficInfo.bounds.max_lon],
        ]
      : ELECTRONIC_CITY_DEFAULT_BOUNDS
    setManualCameraAction({ id: Date.now(), bounds })
  }

  function validateCoordinates(orig, dest) {
    if (!orig || !dest) {
      return 'Please select both a starting location and a destination.'
    }
    const origLat = Number(orig.latitude)
    const origLon = Number(orig.longitude)
    const destLat = Number(dest.latitude)
    const destLon = Number(dest.longitude)

    if (
      !Number.isFinite(origLat) ||
      !Number.isFinite(origLon) ||
      !Number.isFinite(destLat) ||
      !Number.isFinite(destLon) ||
      origLat < -90 ||
      origLat > 90 ||
      destLat < -90 ||
      destLat > 90 ||
      origLon < -180 ||
      origLon > 180 ||
      destLon < -180 ||
      destLon > 180
    ) {
      return 'Invalid coordinates selected. Please re-select origin and destination.'
    }

    const latDiff = Math.abs(origLat - destLat) * 111.32
    const lonDiff = Math.abs(origLon - destLon) * 111.32 * Math.cos((origLat * Math.PI) / 180)
    const approxDistKm = Math.hypot(latDiff, lonDiff)
    if (approxDistKm > 20) {
      return `Selected locations are approximately ${approxDistKm.toFixed(1)} km apart. Pedestrian network downloads are limited to 20 km to ensure high performance.`
    }

    return null
  }

  function formatApiError(error) {
    const msg = error instanceof Error ? error.message : String(error)
    if (msg.includes('503') || msg.includes('temporarily busy') || msg.includes('Overpass')) {
      return 'OpenStreetMap routing service is temporarily at capacity. The system caches downloaded networks; please wait a moment and click "Find and compare routes" again.'
    }
    if (msg.includes('422') || msg.includes('too far apart')) {
      return 'The selected locations are too far apart for a pedestrian route download (maximum 20 km diagonal).'
    }
    if (msg.includes('404') || msg.includes('No pedestrian path')) {
      return 'No walkable pedestrian path connects the selected locations. Try moving pins to adjacent public roads or sidewalks.'
    }
    return msg || 'Unable to calculate routes. Please check connection and retry.'
  }

  // Primary action: Find and compare both routes in one step
  async function handleFindAndCompare() {
    const validationError = validateCoordinates(origin, destination)
    if (validationError) {
      setRouteError(validationError)
      return
    }

    clearRouteRequest()
    const controller = new AbortController()
    requestController.current = controller
    setIsCalculating(true)
    setCalculatingMode('compare')
    setRoute(null)
    setComparison(null)
    setRouteError('')
    setManualCameraAction(null)

    try {
      const result = await compareRoutes(
        { latitude: Number(origin.latitude), longitude: Number(origin.longitude) },
        { latitude: Number(destination.latitude), longitude: Number(destination.longitude) },
        { timestamp: selectedTimestamp, trafficWeight },
        controller.signal,
      )

      if (!controller.signal.aborted) {
        setComparison(result)
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setRouteError(formatApiError(error))
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null
        setIsCalculating(false)
        setCalculatingMode('')
      }
    }
  }

  // Secondary option: Calculate shortest route only
  async function handleCalculateShortestOnly() {
    const validationError = validateCoordinates(origin, destination)
    if (validationError) {
      setRouteError(validationError)
      return
    }

    clearRouteRequest()
    const controller = new AbortController()
    requestController.current = controller
    setIsCalculating(true)
    setCalculatingMode('shortest')
    setRoute(null)
    setComparison(null)
    setRouteError('')
    setManualCameraAction(null)

    try {
      const result = await calculateShortestRoute(
        { latitude: Number(origin.latitude), longitude: Number(origin.longitude) },
        { latitude: Number(destination.latitude), longitude: Number(destination.longitude) },
        controller.signal,
      )

      if (!controller.signal.aborted) {
        setRoute(result)
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setRouteError(formatApiError(error))
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null
        setIsCalculating(false)
        setCalculatingMode('')
      }
    }
  }

  // Difference metrics between comparison routes
  const shortestDist = comparison?.shortest_distance?.distance_meters
  const trafficDist = comparison?.traffic_accessibility_aware?.distance_meters
  const distanceDelta =
    shortestDist !== undefined && trafficDist !== undefined
      ? trafficDist - shortestDist
      : null
  const costDelta =
    comparison?.shortest_distance?.cost !== undefined &&
    comparison?.traffic_accessibility_aware?.cost !== undefined
      ? comparison.traffic_accessibility_aware.cost - comparison.shortest_distance.cost
      : null

  const hasTrafficAlternative = comparison?.status === 'alternative_found'
  const hasMatchedTraffic =
    (comparison?.traffic_accessibility_aware?.matched_traffic_edges_count || 0) > 0
  const hasImpedanceReduction =
    costDelta !== null && costDelta < -0.01 && hasTrafficAlternative && hasMatchedTraffic

  return (
    <div className="dashboard-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon">🚶🚦</div>
          <div>
            <h1>AccessRoute</h1>
            <p className="header-subtitle">
              AI-Based Accessibility & Traffic-Aware Route Recommendation for Public Spaces
            </p>
          </div>
        </div>

        <div className="header-badges">
          <span className="prototype-badge">Research Prototype</span>
          <span className="coverage-badge">
            {trafficInfo?.available
              ? '● Electronic City Traffic Active'
              : '○ Traffic Data Offline'}
          </span>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="dashboard-main-grid">
        {/* Left Column: Planning Controls */}
        <aside className="planning-panel" aria-label="Route Planning Controls">
          {/* Section: Origin and Destination */}
          <div className="panel-card location-card">
            <div className="card-header-row">
              <h2>Location Inputs</h2>
              <button
                type="button"
                className="clear-all-link"
                onClick={handleReset}
                disabled={!origin && !destination}
              >
                Reset
              </button>
            </div>

            <LocationInput
              id="starting-location"
              label="Starting Location (A)"
              placeholder="Search origin address or click map..."
              value={origin}
              onChange={(loc) => {
                clearRouteRequest()
                setRoute(null)
                setComparison(null)
                setManualCameraAction(null)
                setOrigin(loc)
              }}
              onClear={() => {
                clearRouteRequest()
                setRoute(null)
                setComparison(null)
                setManualCameraAction(null)
                setOrigin(null)
              }}
              isActiveTarget={selectionTarget === 'origin'}
              onActivateTarget={() => setSelectionTarget('origin')}
            />

            <LocationInput
              id="destination-location"
              label="Destination (B)"
              placeholder="Search destination address or click map..."
              value={destination}
              onChange={(loc) => {
                clearRouteRequest()
                setRoute(null)
                setComparison(null)
                setManualCameraAction(null)
                setDestination(loc)
              }}
              onClear={() => {
                clearRouteRequest()
                setRoute(null)
                setComparison(null)
                setManualCameraAction(null)
                setDestination(null)
              }}
              isActiveTarget={selectionTarget === 'destination'}
              onActivateTarget={() => setSelectionTarget('destination')}
            />

            {/* Quick preset routes */}
            <div className="preset-routes-block">
              <span className="presets-title">Quick Electronic City sample routes:</span>
              <div className="preset-chips-row">
                {PRESET_ROUTES.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className="preset-chip"
                    onClick={() => handleApplyPreset(p)}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Section: Traffic Settings */}
          <div className="panel-card traffic-card">
            <h2>Traffic Parameters</h2>
            <div className="param-field">
              <label htmlFor="traffic-timestamp-select">
                Traffic time (5-min resolution):
              </label>
              <select
                id="traffic-timestamp-select"
                value={selectedTimestamp}
                onChange={(e) => {
                  clearRouteRequest()
                  setComparison(null)
                  setSelectedTimestamp(e.target.value)
                }}
              >
                {trafficInfo?.sample_presets?.map((preset) => (
                  <option key={preset.id} value={preset.timestamp}>
                    {preset.label}
                  </option>
                )) || (
                  <>
                    <option value="2026-01-01 08:30:00">
                      Morning Peak (08:30 AM)
                    </option>
                    <option value="2026-01-01 13:00:00">
                      Afternoon (01:00 PM)
                    </option>
                    <option value="2026-01-01 18:30:00">
                      Evening Peak (06:30 PM)
                    </option>
                    <option value="2026-01-01 02:00:00">
                      Night / Free-Flow (02:00 AM)
                    </option>
                  </>
                )}
              </select>
              <p className="param-hint">
                Simulated vehicular time series for Electronic City (Jan–Feb 2026).
              </p>
            </div>

            <div className="param-field">
              <div className="param-label-row">
                <label htmlFor="traffic-weight-slider">
                  Traffic Penalty Weight (α = {trafficWeight.toFixed(1)})
                </label>
              </div>
              <input
                id="traffic-weight-slider"
                type="range"
                min="0.0"
                max="5.0"
                step="0.5"
                value={trafficWeight}
                onChange={(e) => {
                  clearRouteRequest()
                  setComparison(null)
                  setTrafficWeight(parseFloat(e.target.value))
                }}
              />
              <p className="param-hint">
                Higher α encourages detours onto quieter paths to bypass congestion. At
                α = 0, route prioritizes pure physical distance.
              </p>
            </div>

            <div className="cost-formula-callout">
              <code>C(e) = d(e) · (1 + α · T(e))</code>
              <p className="formula-note">
                For matched edges, T(e) in [0, 1] is simulated vehicular congestion.
                For unmatched edges, traffic is unknown, so T(e) = 0 is applied (cost = physical length d(e);
                missing traffic values are not measured zero congestion).
              </p>
            </div>
          </div>

          {/* Section: Action Buttons */}
          <div className="action-button-group">
            <button
              type="button"
              className="primary-action-btn"
              onClick={handleFindAndCompare}
              disabled={!origin || !destination || isCalculating}
            >
              {isCalculating && calculatingMode === 'compare' ? (
                <>
                  <span className="btn-spinner" /> Finding & comparing routes…
                </>
              ) : (
                <> Find and compare routes</>
              )}
            </button>

            <button
              type="button"
              className="secondary-action-btn"
              onClick={handleCalculateShortestOnly}
              disabled={!origin || !destination || isCalculating}
            >
              {isCalculating && calculatingMode === 'shortest'
                ? 'Calculating shortest route…'
                : 'Calculate shortest route only'}
            </button>
          </div>

          {/* Error Banner */}
          {routeError && (
            <div className="alert-banner error-banner" role="alert">
              <span className="alert-icon">⚠️</span>
              <div>
                <strong>Routing Notice:</strong> {routeError}
              </div>
            </div>
          )}
        </aside>

        {/* Right Column: Map & Comparison Results */}
        <section className="map-and-results-section" aria-label="Route Map and Analysis">
          {/* Map Utility Toolbar */}
          <div className="map-toolbar">
            <button
              type="button"
              className="nav-ec-button"
              onClick={handleGoToElectronicCity}
              title="Center and fit map to Electronic City dataset bounds"
            >
              Go to Electronic City
            </button>

            <div className="toolbar-info-text">
              <span>Traffic dataset covers 602 directed road edges in Electronic City.</span>
            </div>
          </div>

          {/* Prominent Leaflet Map */}
          <MapView
            origin={origin}
            destination={destination}
            route={route}
            comparison={comparison}
            electronicCityBounds={trafficInfo?.bounds}
            manualCameraAction={manualCameraAction}
            onLocationSelect={handleLocationSelectFromMap}
            selectionTarget={selectionTarget}
          />

          {/* Single Route Result (when Shortest Only was calculated) */}
          {route && !comparison && (
            <div className="single-route-card" aria-live="polite">
              <div className="single-card-header">
                <h3>Shortest Walking Route</h3>
                <span className="route-length-tag">
                  {formatDistance(route.distance_meters)}
                </span>
              </div>
              <p className="single-card-meta">
                Baseline walking route calculated via Dijkstra on OpenStreetMap pedestrian network.
                Preserves ability to calculate pedestrian paths anywhere walkable OpenStreetMap data is available.
              </p>
            </div>
          )}

          {/* Dual Route Comparison Cards */}
          {comparison && (
            <div className="comparison-results-panel" aria-live="polite">
              {/* Header Status Row */}
              <div className="comparison-status-banner">
                <div>
                  <h3 className="comparison-panel-title">Route Comparison Results</h3>
                  <p className="timestamp-badge">
                    Simulated traffic evaluation time: <strong>{comparison.selected_timestamp}</strong>
                  </p>
                </div>
                <span className={`status-pill ${comparison.status}`}>
                  {comparison.status === 'alternative_found' &&
                    '✓ Alternative Route Found (Avoids High Traffic)'}
                  {comparison.status === 'identical_routes' &&
                    'ℹ Shortest Path Is Already Optimal (Minimal Congestion)'}
                  {comparison.status === 'no_traffic_coverage_in_area' &&
                    '⚠️ Outside Traffic Coverage Area (Shortest Walk Fallback)'}
                  {comparison.status === 'traffic_dataset_unavailable' &&
                    '⚠️ Traffic Dataset Offline (Shortest Walk Fallback)'}
                  {comparison.status === 'partial_shortest_only' &&
                    '⚠️ Traffic Alternative Unavailable (Shortest Walk Fallback)'}
                </span>
              </div>

              {/* Explanatory Fallback or Optimal Status Notice */}
              {comparison.status === 'no_traffic_coverage_in_area' && (
                <div className="fallback-explanation-banner">
                  <span className="fallback-banner-icon">ℹ️</span>
                  <div className="fallback-banner-text">
                    <strong>Pedestrian route available; traffic data unavailable for this area.</strong>
                    <p>
                      The walking path was calculated using OpenStreetMap pedestrian data.
                      However, vehicular traffic data is unavailable for road edges in this area
                      (the simulated dataset covers Electronic City only). Because traffic conditions along these
                      edges are unknown, the system falls back to the shortest physical walking route.
                    </p>
                  </div>
                </div>
              )}

              {comparison.status === 'identical_routes' && (
                <div className="identical-explanation-banner">
                  <span className="identical-banner-icon">✓</span>
                  <div className="identical-banner-text">
                    <strong>Shortest walking route is already optimal under simulated conditions.</strong>
                    <p>
                      The shortest walking path encounters minimal traffic impedance.
                      Taking a detour to avoid congestion is not justified, so both route options follow the identical physical path.
                    </p>
                  </div>
                </div>
              )}

              {comparison.status === 'partial_shortest_only' && (
                <div className="fallback-explanation-banner">
                  <span className="fallback-banner-icon">⚠️</span>
                  <div className="fallback-banner-text">
                    <strong>Traffic-aware route calculation could not be completed.</strong>
                    <p>
                      Showing the verified shortest walking route as a safe fallback.
                    </p>
                  </div>
                </div>
              )}

              {comparison.status === 'traffic_dataset_unavailable' && (
                <div className="fallback-explanation-banner">
                  <span className="fallback-banner-icon">⚠️</span>
                  <div className="fallback-banner-text">
                    <strong>Electronic City traffic dataset is currently offline.</strong>
                    <p>
                      Showing the verified shortest walking route as a safe fallback.
                    </p>
                  </div>
                </div>
              )}

              {/* Delta highlights callout - ONLY shown when a genuine traffic alternative exists with validated reduction */}
              {hasImpedanceReduction && (
                <div className="delta-highlight-callout">
                  <span className="delta-icon">💡</span>
                  <span>
                    The traffic-aware route is{' '}
                    <strong>{formatDistance(Math.abs(distanceDelta))} longer</strong>{' '}
                    ({shortestDist ? ((distanceDelta / shortestDist) * 100).toFixed(1) : 0}%),
                    but reduces simulated traffic impedance by{' '}
                    <strong>{Math.abs(costDelta).toFixed(1)} units</strong>{' '}
                    ({comparison.shortest_distance.cost > 0
                      ? ((Math.abs(costDelta) / comparison.shortest_distance.cost) * 100).toFixed(1)
                      : 0}%)
                    based on {comparison.traffic_accessibility_aware.matched_traffic_edges_count} monitored road segments.
                  </span>
                </div>
              )}

              {/* Two Side-by-Side Cards */}
              <div className="comparison-cards-grid">
                {/* Shortest Route Card (BLUE) */}
                <div className="route-result-card shortest-card">
                  <div className="route-card-top">
                    <span className="route-color-indicator blue-ind" />
                    <div>
                      <h4>Shortest Walking Route</h4>
                      <span className="card-badge">Baseline Distance</span>
                    </div>
                  </div>

                  <div className="stat-highlight">
                    <span className="stat-label">Total Distance</span>
                    <span className="stat-value primary-blue">
                      {formatDistance(comparison.shortest_distance.distance_meters)}
                    </span>
                  </div>

                  <dl className="stat-list">
                    <div className="stat-row">
                      <dt>Routing Impedance Cost</dt>
                      <dd>{comparison.shortest_distance.cost.toFixed(1)}</dd>
                    </div>

                    <div className="stat-row">
                      <dt>Traffic Data Coverage</dt>
                      <dd>
                        {comparison.shortest_distance.traffic_coverage_percent}%
                        {comparison.shortest_distance.traffic_coverage_percent === 0 ? (
                          <span className="coverage-subhint">
                            0 of {comparison.shortest_distance.edges_count} edges (No traffic data for area; traffic conditions unknown)
                          </span>
                        ) : comparison.shortest_distance.traffic_coverage_percent < 100 ? (
                          <span className="coverage-subhint">
                            {comparison.shortest_distance.matched_traffic_edges_count} of{' '}
                            {comparison.shortest_distance.edges_count} edges matched ({comparison.shortest_distance.edges_count - comparison.shortest_distance.matched_traffic_edges_count} unmatched edges have unknown traffic)
                          </span>
                        ) : (
                          <span className="coverage-subhint" style={{ color: '#166534' }}>
                            {comparison.shortest_distance.matched_traffic_edges_count} of{' '}
                            {comparison.shortest_distance.edges_count} edges (Full dataset coverage)
                          </span>
                        )}
                      </dd>
                    </div>

                    <div className="stat-row">
                      <dt>Average Congestion</dt>
                      <dd>
                        {comparison.shortest_distance.average_congestion !== null ? (
                          <>
                            {(comparison.shortest_distance.average_congestion * 100).toFixed(1)}%
                            <span className="congestion-unknown-note">
                              Measured across {comparison.shortest_distance.matched_traffic_edges_count} monitored segment(s)
                            </span>
                          </>
                        ) : (
                          <>
                            Unknown
                            <span className="congestion-unknown-note">
                              No monitored road segments; missing values are not zero congestion
                            </span>
                          </>
                        )}
                      </dd>
                    </div>

                    <div className="stat-row">
                      <dt>Network Segments</dt>
                      <dd>{comparison.shortest_distance.edges_count} edges</dd>
                    </div>
                  </dl>
                </div>

                {/* Traffic-Aware Alternative Card (GREEN) */}
                <div className="route-result-card traffic-card">
                  <div className="route-card-top">
                    <span className="route-color-indicator green-ind" />
                    <div>
                      {hasTrafficAlternative ? (
                        <>
                          <h4>Traffic-Aware Alternative</h4>
                          <span className="card-badge">Congestion-Averse Detour</span>
                        </>
                      ) : comparison.status === 'identical_routes' ? (
                        <>
                          <h4>Traffic-Aware Route</h4>
                          <span className="card-badge optimal-badge">Identical to Shortest Walk</span>
                        </>
                      ) : (
                        <>
                          <h4>Traffic-Aware Route (Fallback)</h4>
                          <span className="card-badge fallback-badge">Shortest Walk Fallback</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="stat-highlight">
                    <span className="stat-label">Total Distance</span>
                    <span className="stat-value primary-green">
                      {formatDistance(comparison.traffic_accessibility_aware.distance_meters)}
                      {hasTrafficAlternative && distanceDelta !== null && distanceDelta !== 0 && (
                        <span className="delta-subtext">
                          {distanceDelta > 0 ? ` (+${formatDistance(distanceDelta)})` : ` (${formatDistance(distanceDelta)})`}
                        </span>
                      )}
                      {!hasTrafficAlternative && (
                        <span className="delta-subtext"> (Identical)</span>
                      )}
                    </span>
                  </div>

                  <dl className="stat-list">
                    <div className="stat-row">
                      <dt>Routing Impedance Cost</dt>
                      <dd>{comparison.traffic_accessibility_aware.cost.toFixed(1)}</dd>
                    </div>

                    <div className="stat-row">
                      <dt>Traffic Data Coverage</dt>
                      <dd>
                        {comparison.traffic_accessibility_aware.traffic_coverage_percent}%
                        {comparison.traffic_accessibility_aware.traffic_coverage_percent === 0 ? (
                          <span className="coverage-subhint">
                            0 of {comparison.traffic_accessibility_aware.edges_count} edges (No traffic data for area; traffic conditions unknown)
                          </span>
                        ) : comparison.traffic_accessibility_aware.traffic_coverage_percent < 100 ? (
                          <span className="coverage-subhint">
                            {comparison.traffic_accessibility_aware.matched_traffic_edges_count} of{' '}
                            {comparison.traffic_accessibility_aware.edges_count} edges matched ({comparison.traffic_accessibility_aware.edges_count - comparison.traffic_accessibility_aware.matched_traffic_edges_count} unmatched edges have unknown traffic)
                          </span>
                        ) : (
                          <span className="coverage-subhint" style={{ color: '#166534' }}>
                            {comparison.traffic_accessibility_aware.matched_traffic_edges_count} of{' '}
                            {comparison.traffic_accessibility_aware.edges_count} edges (Full dataset coverage)
                          </span>
                        )}
                      </dd>
                    </div>

                    <div className="stat-row">
                      <dt>Average Congestion</dt>
                      <dd>
                        {comparison.traffic_accessibility_aware.average_congestion !== null ? (
                          <>
                            {(comparison.traffic_accessibility_aware.average_congestion * 100).toFixed(1)}%
                            <span className="congestion-unknown-note">
                              Measured across {comparison.traffic_accessibility_aware.matched_traffic_edges_count} monitored segment(s)
                            </span>
                          </>
                        ) : (
                          <>
                            Unknown
                            <span className="congestion-unknown-note">
                              No monitored road segments; missing values are not zero congestion
                            </span>
                          </>
                        )}
                      </dd>
                    </div>

                    <div className="stat-row">
                      <dt>Network Segments</dt>
                      <dd>{comparison.traffic_accessibility_aware.edges_count} edges</dd>
                    </div>
                  </dl>
                </div>
              </div>

              {/* Research & Disclaimer Notice */}
              <div className="disclaimer-callout">
                <span className="disclaimer-icon">⚠️</span>
                <p>
                  <strong>Academic Research Disclaimer:</strong> {comparison.disclaimer}
                  <br />
                  Traffic metrics represent simulated vehicular volume and flow from the
                  Electronic City dataset. They do not represent real-time sensor measurements,
                  live traffic, or certified pedestrian sidewalk accessibility.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
