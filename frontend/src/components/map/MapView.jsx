import { useEffect, useRef } from 'react'
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Rectangle,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import { Compass, MapPin } from 'lucide-react'
import { DEFAULT_MAP_CENTER } from './mapConfig'
import './MapView.css'

function LocationSelector({ onSelect }) {
  useMapEvents({
    click(event) {
      onSelect({
        latitude: event.latlng.lat,
        longitude: event.latlng.lng,
      })
    },
  })

  return null
}

function MapController({
  origin,
  destination,
  route,
  comparison,
  manualCameraAction,
  onMapReady,
}) {
  const map = useMap()
  const lastTargetKeyRef = useRef(null)

  useEffect(() => {
    if (onMapReady) {
      onMapReady(map)
    }
  }, [map, onMapReady])

  // Invalidate map size on layout changes or window resizes
  useEffect(() => {
    const container = map.getContainer()
    if (!container) return

    map.invalidateSize()

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => {
        map.invalidateSize()
      })
      observer.observe(container)
      return () => observer.disconnect()
    }
  }, [map, route, comparison])

  useEffect(() => {
    // 1. Manual Camera Action (e.g. "Go to Electronic City")
    if (manualCameraAction?.bounds) {
      const key = `manual:${manualCameraAction.id || JSON.stringify(manualCameraAction.bounds)}`
      if (lastTargetKeyRef.current !== key) {
        lastTargetKeyRef.current = key
        try {
          map.fitBounds(manualCameraAction.bounds, {
            padding: [40, 40],
            maxZoom: 16,
            animate: true,
            duration: 0.8,
          })
        } catch {
          // safe fallback
        }
      }
      return
    }

    // 2. Comparison Result (fits both route geometries & endpoints)
    if (comparison) {
      const allCoords = [
        ...(comparison.shortest_distance?.coordinates || []),
        ...(comparison.traffic_accessibility_aware?.coordinates || []),
      ]
      if (allCoords.length > 0) {
        const key = `comp:${comparison.status}:${allCoords.length}:${allCoords[0].latitude.toFixed(4)},${allCoords[0].longitude.toFixed(4)}->${allCoords[allCoords.length - 1].latitude.toFixed(4)},${allCoords[allCoords.length - 1].longitude.toFixed(4)}`
        if (lastTargetKeyRef.current !== key) {
          lastTargetKeyRef.current = key
          const lats = allCoords.map((c) => c.latitude)
          const lons = allCoords.map((c) => c.longitude)
          const minLat = Math.min(...lats)
          const maxLat = Math.max(...lats)
          const minLon = Math.min(...lons)
          const maxLon = Math.max(...lons)

          try {
            if (Math.abs(maxLat - minLat) < 0.0001 && Math.abs(maxLon - minLon) < 0.0001) {
              map.setView([minLat, minLon], 16, { animate: true, duration: 0.8 })
            } else {
              map.fitBounds(
                [
                  [minLat, minLon],
                  [maxLat, maxLon],
                ],
                {
                  padding: [45, 45],
                  maxZoom: 16,
                  animate: true,
                  duration: 0.8,
                },
              )
            }
          } catch {
            // safe fallback
          }
        }
        return
      }
    }

    // 3. Single Shortest Route Result
    if (route && route.coordinates?.length > 0) {
      const coords = route.coordinates
      const key = `route:${coords.length}:${coords[0].latitude.toFixed(4)},${coords[0].longitude.toFixed(4)}->${coords[coords.length - 1].latitude.toFixed(4)},${coords[coords.length - 1].longitude.toFixed(4)}`
      if (lastTargetKeyRef.current !== key) {
        lastTargetKeyRef.current = key
        const lats = coords.map((c) => c.latitude)
        const lons = coords.map((c) => c.longitude)
        const minLat = Math.min(...lats)
        const maxLat = Math.max(...lats)
        const minLon = Math.min(...lons)
        const maxLon = Math.max(...lons)

        try {
          if (Math.abs(maxLat - minLat) < 0.0001 && Math.abs(maxLon - minLon) < 0.0001) {
            map.setView([minLat, minLon], 16, { animate: true, duration: 0.8 })
          } else {
            map.fitBounds(
              [
                [minLat, minLon],
                [maxLat, maxLon],
              ],
              {
                padding: [45, 45],
                maxZoom: 16,
                animate: true,
                duration: 0.8,
              },
            )
          }
        } catch {
          // safe fallback
        }
      }
      return
    }

    // 4. Both Origin and Destination selected
    if (origin && destination) {
      const origLat = Number(origin.latitude)
      const origLon = Number(origin.longitude)
      const destLat = Number(destination.latitude)
      const destLon = Number(destination.longitude)

      if (
        Number.isFinite(origLat) &&
        Number.isFinite(origLon) &&
        Number.isFinite(destLat) &&
        Number.isFinite(destLon)
      ) {
        const key = `both:${origLat.toFixed(5)},${origLon.toFixed(5)}|${destLat.toFixed(5)},${destLon.toFixed(5)}`
        if (lastTargetKeyRef.current !== key) {
          lastTargetKeyRef.current = key
          const minLat = Math.min(origLat, destLat)
          const maxLat = Math.max(origLat, destLat)
          const minLon = Math.min(origLon, destLon)
          const maxLon = Math.max(origLon, destLon)

          try {
            if (Math.abs(maxLat - minLat) < 0.0001 && Math.abs(maxLon - minLon) < 0.0001) {
              map.setView([minLat, minLon], 16, { animate: true, duration: 0.8 })
            } else {
              map.fitBounds(
                [
                  [minLat, minLon],
                  [maxLat, maxLon],
                ],
                {
                  padding: [50, 50],
                  maxZoom: 16,
                  animate: true,
                  duration: 0.8,
                },
              )
            }
          } catch {
            // safe fallback
          }
        }
      }
      return
    }

    // 5. Only Origin is selected
    if (origin && !destination) {
      const origLat = Number(origin.latitude)
      const origLon = Number(origin.longitude)
      if (Number.isFinite(origLat) && Number.isFinite(origLon)) {
        const key = `origin:${origLat.toFixed(5)},${origLon.toFixed(5)}`
        if (lastTargetKeyRef.current !== key) {
          lastTargetKeyRef.current = key
          try {
            map.setView([origLat, origLon], 16, { animate: true, duration: 0.8 })
          } catch {
            // safe fallback
          }
        }
      }
      return
    }

    // 6. Only Destination is selected
    if (!origin && destination) {
      const destLat = Number(destination.latitude)
      const destLon = Number(destination.longitude)
      if (Number.isFinite(destLat) && Number.isFinite(destLon)) {
        const key = `dest:${destLat.toFixed(5)},${destLon.toFixed(5)}`
        if (lastTargetKeyRef.current !== key) {
          lastTargetKeyRef.current = key
          try {
            map.setView([destLat, destLon], 16, { animate: true, duration: 0.8 })
          } catch {
            // safe fallback
          }
        }
      }
      return
    }

    // 7. Neither location selected
    if (!origin && !destination && !route && !comparison) {
      lastTargetKeyRef.current = null
    }
  }, [map, origin, destination, route, comparison, manualCameraAction])

  return null
}

function MapLocationMarker({ label, location, color, badgeText }) {
  if (!location) return null

  return (
    <CircleMarker
      center={[location.latitude, location.longitude]}
      radius={10}
      pathOptions={{
        color: '#ffffff',
        fillColor: color,
        fillOpacity: 1,
        weight: 3,
      }}
    >
      <Tooltip permanent direction="top" offset={[0, -10]}>
        <strong>{badgeText}</strong> {label}
      </Tooltip>
    </CircleMarker>
  )
}

function getMapStatus(origin, destination, route, comparison) {
  if (comparison) {
    if (comparison.status === 'alternative_found') {
      return {
        variant: 'comparison',
        text: 'Showing comparison: shortest vs traffic-aware routes',
      }
    }
    if (comparison.status === 'identical_routes') {
      return {
        variant: 'identical',
        text: 'Showing identical routes (shortest path is optimal)',
      }
    }
    if (comparison.status === 'no_traffic_coverage_in_area') {
      return {
        variant: 'fallback',
        text: 'Showing shortest route (outside traffic coverage area)',
      }
    }
    if (comparison.status === 'traffic_dataset_unavailable') {
      return {
        variant: 'fallback',
        text: 'Showing shortest route (traffic dataset offline)',
      }
    }
    if (comparison.status === 'partial_shortest_only') {
      return {
        variant: 'fallback',
        text: 'Showing shortest route (traffic alternative unavailable)',
      }
    }
    return {
      variant: 'comparison',
      text: 'Showing calculated comparison routes',
    }
  }

  if (route) {
    return {
      variant: 'route',
      text: 'Showing calculated shortest walking route',
    }
  }

  if (origin && destination) {
    return {
      variant: 'both',
      text: 'Both endpoints selected',
    }
  }

  if (origin && !destination) {
    return {
      variant: 'origin',
      text: `Starting point: ${origin.label || 'Point A'}`,
    }
  }

  if (!origin && destination) {
    return {
      variant: 'destination',
      text: `Destination: ${destination.label || 'Point B'}`,
    }
  }

  return {
    variant: 'idle',
    text: 'Click map or search to choose locations',
  }
}

export default function MapView({
  origin,
  destination,
  route,
  comparison,
  electronicCityBounds,
  manualCameraAction,
  onLocationSelect,
  selectionTarget,
  onGoToElectronicCity,
  onMapReady,
}) {
  const shortestCoords = comparison?.shortest_distance?.coordinates?.map(
    ({ latitude, longitude }) => [latitude, longitude],
  )
  const trafficCoords =
    comparison?.traffic_accessibility_aware?.coordinates?.map(
      ({ latitude, longitude }) => [latitude, longitude],
    )

  const ecRectBounds = electronicCityBounds
    ? [
        [electronicCityBounds.min_lat, electronicCityBounds.min_lon],
        [electronicCityBounds.max_lat, electronicCityBounds.max_lon],
      ]
    : null

  const statusInfo = getMapStatus(origin, destination, route, comparison)
  const isAlternativeFound = comparison?.status === 'alternative_found'

  return (
    <div className="map-wrapper relative w-full overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-xs">
      <MapContainer
        className="route-map"
        center={DEFAULT_MAP_CENTER}
        zoom={13}
        scrollWheelZoom
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapController
          origin={origin}
          destination={destination}
          route={route}
          comparison={comparison}
          manualCameraAction={manualCameraAction}
          onMapReady={onMapReady}
        />
        <LocationSelector onSelect={onLocationSelect} />

        {/* Electronic City simulated dataset coverage boundary */}
        {ecRectBounds && (
          <Rectangle
            bounds={ecRectBounds}
            pathOptions={{
              color: '#d97706',
              weight: 1.5,
              dashArray: '6, 6',
              fillColor: '#fef3c7',
              fillOpacity: 0.08,
            }}
          >
            <Tooltip sticky>Electronic City Traffic Dataset Coverage Area</Tooltip>
          </Rectangle>
        )}

        {/* Route A: Shortest walking route (BLUE) */}
        {comparison && shortestCoords && (
          <Polyline
            positions={shortestCoords}
            pathOptions={{
              color: '#2563EB',
              weight: 6,
              opacity: 0.9,
            }}
          />
        )}

        {/* Route B: Traffic-aware alternative (GREEN) - rendered only when a genuine alternative route exists */}
        {comparison && trafficCoords && isAlternativeFound && (
          <Polyline
            positions={trafficCoords}
            pathOptions={{
              color: '#15803D',
              weight: 3.5,
              dashArray: '8, 8',
              opacity: 1.0,
            }}
          />
        )}

        {/* Single shortest route fallback */}
        {!comparison && route && (
          <Polyline
            positions={route.coordinates.map(({ latitude, longitude }) => [
              latitude,
              longitude,
            ])}
            pathOptions={{ color: '#2563EB', weight: 6, opacity: 0.9 }}
          />
        )}

        {/* Snapped endpoints indicators */}
        {comparison?.snapped_origin && (
          <CircleMarker
            center={[
              comparison.snapped_origin.latitude,
              comparison.snapped_origin.longitude,
            ]}
            radius={4}
            pathOptions={{ color: '#16a34a', fillColor: '#16a34a', fillOpacity: 0.8 }}
          />
        )}
        {comparison?.snapped_destination && (
          <CircleMarker
            center={[
              comparison.snapped_destination.latitude,
              comparison.snapped_destination.longitude,
            ]}
            radius={4}
            pathOptions={{ color: '#dc2626', fillColor: '#dc2626', fillOpacity: 0.8 }}
          />
        )}

        {/* Origin Marker (Green) */}
        <MapLocationMarker
          badgeText="A"
          label={origin?.label || 'Origin'}
          location={origin}
          color="#16a34a"
        />

        {/* Destination Marker (Red) */}
        <MapLocationMarker
          badgeText="B"
          label={destination?.label || 'Destination'}
          location={destination}
          color="#dc2626"
        />
      </MapContainer>

      {/* Floating Status Pill (Top-Left) */}
      <div className="absolute top-2.5 left-2.5 z-[1000] pointer-events-none max-w-[calc(100%-140px)] sm:max-w-xs">
        <div className="pointer-events-auto flex items-center gap-1.5 sm:gap-2 rounded-full border border-slate-200/90 bg-white/95 px-2.5 py-1 sm:px-3 sm:py-1.5 text-[11px] sm:text-xs font-semibold text-slate-800 shadow-sm backdrop-blur-md min-w-0">
          <span className={`h-2 w-2 rounded-full shrink-0 ${
            statusInfo.variant === 'comparison'
              ? 'bg-emerald-500 ring-2 ring-emerald-200'
              : statusInfo.variant === 'route' || statusInfo.variant === 'identical'
              ? 'bg-blue-500 ring-2 ring-blue-200'
              : statusInfo.variant === 'fallback'
              ? 'bg-amber-500 ring-2 ring-amber-200'
              : statusInfo.variant === 'origin'
              ? 'bg-emerald-500 ring-2 ring-emerald-200'
              : statusInfo.variant === 'destination'
              ? 'bg-rose-500 ring-2 ring-rose-200'
              : 'bg-slate-400'
          }`} />
          <span className="truncate">{statusInfo.text}</span>
        </div>
      </div>

      {/* Floating Legend & Controls (Top-Right) */}
      <div className="absolute top-2.5 right-2.5 z-[1000] flex flex-col items-end gap-1.5 pointer-events-none">
        {/* Go to Electronic City action button */}
        {onGoToElectronicCity && (
          <button
            type="button"
            onClick={onGoToElectronicCity}
            className="pointer-events-auto flex items-center gap-1.5 rounded-lg border border-slate-200/90 bg-white/95 px-2 py-0.5 sm:px-2.5 sm:py-1 text-[11px] sm:text-xs font-medium text-slate-700 shadow-sm backdrop-blur-md hover:bg-slate-50 transition-colors"
            title="Focus map on Electronic City dataset area"
          >
            <Compass className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-blue-600 shrink-0" />
            <span className="hidden sm:inline">Go to Electronic City</span>
            <span className="sm:hidden">Focus</span>
          </button>
        )}

        {/* Legend */}
        <div className="pointer-events-auto flex flex-col gap-1 sm:gap-1.5 rounded-xl border border-slate-200/90 bg-white/95 p-2 sm:p-2.5 text-[10px] sm:text-[11px] text-slate-700 shadow-sm backdrop-blur-md">
          <div className="flex items-center gap-1.5 sm:gap-2">
            <span className="h-1 w-3 sm:w-4 rounded-sm bg-blue-600 shrink-0" />
            <span>Shortest path</span>
          </div>
          {isAlternativeFound && (
            <div className="flex items-center gap-1.5 sm:gap-2">
              <span className="h-0 w-3 sm:w-4 border-t-2 border-dashed border-emerald-600 shrink-0" />
              <span>Traffic detour</span>
            </div>
          )}
          {comparison && !isAlternativeFound && (
            <div className="flex items-center gap-1.5 sm:gap-2">
              <span className="h-1 w-3 sm:w-4 rounded-sm bg-slate-400 shrink-0" />
              <span className="text-slate-500">Identical path</span>
            </div>
          )}
          {ecRectBounds && (
            <div className="flex items-center gap-1.5 sm:gap-2">
              <span className="h-2 w-3 sm:h-2.5 sm:w-3.5 rounded-xs border border-dashed border-amber-600 bg-amber-100/60 shrink-0" />
              <span className="text-slate-600">Electronic City</span>
            </div>
          )}
        </div>

        {/* Map-click prompt badge (Visible on larger screens) */}
        <div className="pointer-events-auto hidden sm:flex items-center gap-1.5 rounded-lg border border-slate-200/90 bg-white/95 px-2.5 py-1 text-[11px] font-medium text-slate-700 shadow-sm backdrop-blur-md">
          <MapPin className="h-3 w-3 text-slate-400" />
          <span>Click map:</span>
          <span className={`font-semibold ${selectionTarget === 'origin' ? 'text-emerald-700' : 'text-rose-700'}`}>
            {selectionTarget === 'origin' ? 'Origin (A)' : 'Destination (B)'}
          </span>
        </div>
      </div>
    </div>
  )
}
