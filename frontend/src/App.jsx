import { useEffect, useRef, useState } from 'react'
import {
  Navigation,
  Menu,
  Compass,
  ArrowRight,
  SlidersHorizontal,
} from 'lucide-react'
import PlanningControls from './components/PlanningControls'
import MobileDrawer from './components/MobileDrawer'
import MapView from './components/map/MapView'
import RouteResultCards from './components/RouteResultCards'
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
  const [isMobileDrawerOpen, setIsMobileDrawerOpen] = useState(false)

  // Traffic settings
  const [trafficInfo, setTrafficInfo] = useState(null)
  const [selectedTimestamp, setSelectedTimestamp] = useState('2026-01-01 08:30:00')
  const [trafficWeight, setTrafficWeight] = useState(2.0)

  const requestController = useRef(null)
  const mapInstanceRef = useRef(null)

  // Fetch dataset info on mount
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
        // Safe fallback
      })

    return () => controller.abort()
  }, [])

  function invalidateMap() {
    setTimeout(() => {
      mapInstanceRef.current?.invalidateSize()
    }, 280)
  }

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
      setSelectionTarget('destination')
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
    setIsMobileDrawerOpen(false)
    invalidateMap()
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
      return 'OpenStreetMap routing service is temporarily at capacity. The system caches downloaded networks; please wait a moment and try again.'
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
    setIsMobileDrawerOpen(false)
    invalidateMap()

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
    setIsMobileDrawerOpen(false)
    invalidateMap()

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

  const planningProps = {
    origin,
    setOrigin,
    destination,
    setDestination,
    selectionTarget,
    setSelectionTarget,
    onReset: handleReset,
    presetRoutes: PRESET_ROUTES,
    onApplyPreset: handleApplyPreset,
    selectedTimestamp,
    setSelectedTimestamp,
    trafficWeight,
    setTrafficWeight,
    trafficInfo,
    onFindAndCompare: handleFindAndCompare,
    onCalculateShortestOnly: handleCalculateShortestOnly,
    isCalculating,
    calculatingMode,
    routeError,
    onGoToElectronicCity: handleGoToElectronicCity,
    clearRouteRequest,
    setRoute,
    setComparison,
    setManualCameraAction,
  }

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800 flex flex-col font-sans">
      {/* App Header (Responsive: compact on mobile, refined on desktop) */}
      <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/95 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-3.5 sm:px-6">
          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-600 text-white shadow-xs">
              <Navigation className="h-4 w-4" />
            </div>
            <div>
              <span className="text-base font-bold tracking-tight text-slate-900">
                AccessRoute
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs text-slate-500">
                Walking Route Recommendation
              </span>
            </div>
          </div>

          {/* Right Header: Badges & Controls */}
          <div className="flex items-center gap-2">
            {/* Status indicator */}
            <div className="flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600">
              <span
                className={`h-2 w-2 rounded-full ${
                  trafficInfo?.available
                    ? 'bg-emerald-500 ring-2 ring-emerald-200'
                    : 'bg-slate-400'
                }`}
              />
              <span className="hidden sm:inline">
                {trafficInfo?.available
                  ? 'Electronic City Traffic Active'
                  : 'Traffic Data Offline'}
              </span>
              <span className="sm:hidden">
                {trafficInfo?.available ? 'Traffic Active' : 'Offline'}
              </span>
            </div>

            {/* Go to Electronic City button (Desktop) */}
            <button
              type="button"
              onClick={handleGoToElectronicCity}
              className="hidden md:flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors shadow-2xs"
            >
              <Compass className="h-3.5 w-3.5 text-blue-600" />
              <span>Go to Electronic City</span>
            </button>

            {/* Hamburger button (Mobile only) */}
            <button
              type="button"
              onClick={() => {
                setIsMobileDrawerOpen(true)
                invalidateMap()
              }}
              className="relative flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 lg:hidden shadow-2xs"
              aria-label="Open planning menu"
            >
              <Menu className="h-4 w-4" />
              {(origin || destination) && (
                <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-600" />
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Responsive Layout */}
      <main className="flex-1 mx-auto w-full max-w-7xl p-3 sm:p-5 lg:p-6">
        {/* Desktop Layout: 2 Columns */}
        <div className="hidden lg:grid lg:grid-cols-[380px_1fr] xl:grid-cols-[410px_1fr] gap-6 items-start">
          {/* Left Column: Planning Controls */}
          <aside aria-label="Route Planning Controls">
            <PlanningControls {...planningProps} />
          </aside>

          {/* Right Column: Map & Route Results */}
          <section className="space-y-4" aria-label="Route Map and Analysis">
            <MapView
              origin={origin}
              destination={destination}
              route={route}
              comparison={comparison}
              electronicCityBounds={trafficInfo?.bounds}
              manualCameraAction={manualCameraAction}
              onLocationSelect={handleLocationSelectFromMap}
              selectionTarget={selectionTarget}
              onGoToElectronicCity={handleGoToElectronicCity}
              onMapReady={(map) => {
                mapInstanceRef.current = map
              }}
            />

            <RouteResultCards comparison={comparison} route={route} />
          </section>
        </div>

        {/* Mobile / Tablet Map-First Layout (< lg) */}
        <div className="lg:hidden space-y-3">
          {/* Map is at the TOP on mobile */}
          <MapView
            origin={origin}
            destination={destination}
            route={route}
            comparison={comparison}
            electronicCityBounds={trafficInfo?.bounds}
            manualCameraAction={manualCameraAction}
            onLocationSelect={handleLocationSelectFromMap}
            selectionTarget={selectionTarget}
            onGoToElectronicCity={handleGoToElectronicCity}
            onMapReady={(map) => {
              mapInstanceRef.current = map
            }}
          />

          {/* Compact Mobile Quick Action Bar */}
          <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white p-2.5 shadow-xs">
            <button
              type="button"
              onClick={() => {
                setIsMobileDrawerOpen(true)
                invalidateMap()
              }}
              className="flex-1 flex items-center gap-2 text-left px-2 py-1 rounded-lg hover:bg-slate-50 transition-colors"
            >
              <div className="flex flex-col min-w-0">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Endpoints
                </span>
                <span className="text-xs font-semibold text-slate-800 truncate">
                  {origin ? origin.label : 'Select starting point (A)'} ➜{' '}
                  {destination ? destination.label : 'Select destination (B)'}
                </span>
              </div>
            </button>

            <button
              type="button"
              onClick={() => {
                if (origin && destination) {
                  handleFindAndCompare()
                } else {
                  setIsMobileDrawerOpen(true)
                  invalidateMap()
                }
              }}
              className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-xs hover:bg-blue-700 active:scale-[0.98] shrink-0"
            >
              {origin && destination ? (
                <>
                  <span>Compare</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </>
              ) : (
                <>
                  <SlidersHorizontal className="h-3.5 w-3.5" />
                  <span>Plan</span>
                </>
              )}
            </button>
          </div>

          {/* Route results immediately below map on mobile */}
          <RouteResultCards comparison={comparison} route={route} />
        </div>
      </main>

      {/* Mobile Drawer (Slide-over sheet with controls) */}
      <MobileDrawer
        isOpen={isMobileDrawerOpen}
        onClose={() => {
          setIsMobileDrawerOpen(false)
          invalidateMap()
        }}
      >
        <PlanningControls {...planningProps} />
      </MobileDrawer>
    </div>
  )
}
