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
  const [targetBounds, setTargetBounds] = useState(null)

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
  }

  function handleApplyPreset(preset) {
    clearRouteRequest()
    setOrigin(preset.origin)
    setDestination(preset.destination)
    setRoute(null)
    setComparison(null)
    setRouteError('')

    // Fit map to show both preset locations
    const minLat = Math.min(preset.origin.latitude, preset.destination.latitude)
    const maxLat = Math.max(preset.origin.latitude, preset.destination.latitude)
    const minLon = Math.min(preset.origin.longitude, preset.destination.longitude)
    const maxLon = Math.max(preset.origin.longitude, preset.destination.longitude)
    setTargetBounds([
      [minLat, minLon],
      [maxLat, maxLon],
    ])
  }

  function handleGoToElectronicCity() {
    if (trafficInfo?.bounds) {
      setTargetBounds([
        [trafficInfo.bounds.min_lat, trafficInfo.bounds.min_lon],
        [trafficInfo.bounds.max_lat, trafficInfo.bounds.max_lon],
      ])
    } else {
      setTargetBounds(ELECTRONIC_CITY_DEFAULT_BOUNDS)
    }
  }

  // Primary action: Find and compare both routes in one step
  async function handleFindAndCompare() {
    if (!origin || !destination) {
      setRouteError('Please select both a starting location and a destination.')
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

    try {
      const result = await compareRoutes(
        { latitude: origin.latitude, longitude: origin.longitude },
        { latitude: destination.latitude, longitude: destination.longitude },
        { timestamp: selectedTimestamp, trafficWeight },
        controller.signal,
      )

      if (!controller.signal.aborted) {
        setComparison(result)

        // Fit map bounds to encompass both returned routes
        const allCoords = [
          ...(result.shortest_distance?.coordinates || []),
          ...(result.traffic_accessibility_aware?.coordinates || []),
        ]

        if (allCoords.length > 0) {
          const lats = allCoords.map((c) => c.latitude)
          const lons = allCoords.map((c) => c.longitude)
          setTargetBounds([
            [Math.min(...lats), Math.min(...lons)],
            [Math.max(...lats), Math.max(...lons)],
          ])
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setRouteError(
          error instanceof Error ? error.message : 'Unable to calculate routes.',
        )
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
    if (!origin || !destination) {
      setRouteError('Please select both a starting location and a destination.')
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

    try {
      const result = await calculateShortestRoute(
        { latitude: origin.latitude, longitude: origin.longitude },
        { latitude: destination.latitude, longitude: destination.longitude },
        controller.signal,
      )

      if (!controller.signal.aborted) {
        setRoute(result)
        if (result.coordinates?.length > 0) {
          const lats = result.coordinates.map((c) => c.latitude)
          const lons = result.coordinates.map((c) => c.longitude)
          setTargetBounds([
            [Math.min(...lats), Math.min(...lons)],
            [Math.max(...lats), Math.max(...lons)],
          ])
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setRouteError(
          error instanceof Error ? error.message : 'Unable to calculate route.',
        )
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
                setOrigin(loc)
              }}
              onClear={() => {
                clearRouteRequest()
                setRoute(null)
                setComparison(null)
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
                setDestination(loc)
              }}
              onClear={() => {
                clearRouteRequest()
                setRoute(null)
                setComparison(null)
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
                Physical sidewalk accessibility observations (A(e) from computer vision)
                are under development and currently inactive.
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
                <>🚀 Find and compare routes</>
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
              🎯 Go to Electronic City
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
            targetBounds={targetBounds}
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
                    Traffic evaluation timestamp: <strong>{comparison.selected_timestamp}</strong>
                  </p>
                </div>
                <span className={`status-pill ${comparison.status}`}>
                  {comparison.status === 'alternative_found' &&
                    '✓ Alternative Route Found (Avoids High Traffic)'}
                  {comparison.status === 'identical_routes' &&
                    'ℹ Shortest Path Is Already Optimal (Minimal Traffic)'}
                  {comparison.status === 'no_traffic_coverage_in_area' &&
                    '⚠️ Outside Traffic Coverage Area (Defaulted to Shortest)'}
                  {comparison.status === 'traffic_dataset_unavailable' &&
                    '⚠️ Traffic Dataset Unavailable'}
                </span>
              </div>

              {/* Delta highlights callout */}
              {distanceDelta !== null && comparison.status === 'alternative_found' && (
                <div className="delta-highlight-callout">
                  <span className="delta-icon">💡</span>
                  <span>
                    The traffic-aware route is{' '}
                    <strong>{formatDistance(Math.abs(distanceDelta))} longer</strong>{' '}
                    ({shortestDist ? ((distanceDelta / shortestDist) * 100).toFixed(1) : 0}%),
                    but reduces simulated traffic impedance by{' '}
                    <strong>{Math.abs(costDelta || 0).toFixed(0)} units</strong>.
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
                      <dd>{comparison.shortest_distance.traffic_coverage_percent}%</dd>
                    </div>
                    {comparison.shortest_distance.average_congestion !== null && (
                      <div className="stat-row">
                        <dt>Average Congestion</dt>
                        <dd>
                          {(comparison.shortest_distance.average_congestion * 100).toFixed(1)}%
                        </dd>
                      </div>
                    )}
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
                      <h4>Traffic-Aware Alternative</h4>
                      <span className="card-badge">Congestion-Averse</span>
                    </div>
                  </div>

                  <div className="stat-highlight">
                    <span className="stat-label">Total Distance</span>
                    <span className="stat-value primary-green">
                      {formatDistance(comparison.traffic_accessibility_aware.distance_meters)}
                      {distanceDelta !== null && distanceDelta !== 0 && (
                        <span className="delta-subtext">
                          {distanceDelta > 0 ? ` (+${formatDistance(distanceDelta)})` : ` (${formatDistance(distanceDelta)})`}
                        </span>
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
                      <dd>{comparison.traffic_accessibility_aware.traffic_coverage_percent}%</dd>
                    </div>
                    {comparison.traffic_accessibility_aware.average_congestion !== null && (
                      <div className="stat-row">
                        <dt>Average Congestion</dt>
                        <dd>
                          {(comparison.traffic_accessibility_aware.average_congestion * 100).toFixed(1)}%
                        </dd>
                      </div>
                    )}
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
                  Electronic City dataset. They do not certify sidewalk accessibility,
                  pavement condition, curb ramp presence, or pedestrian safety guarantees.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
