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
}) {
  const map = useMap()
  const lastTargetKeyRef = useRef(null)

  // Invalidate map size on mount, resize, or layout changes (e.g. results card toggle)
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
    // 1. Manual Camera Override (e.g. clicking "Go to Electronic City")
    if (manualCameraAction?.bounds) {
      const key = `manual:${manualCameraAction.id || JSON.stringify(manualCameraAction.bounds)}`
      if (lastTargetKeyRef.current !== key) {
        lastTargetKeyRef.current = key
        try {
          map.fitBounds(manualCameraAction.bounds, {
            padding: [45, 45],
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

    // 2. Comparison Result (fits both shortest and traffic routes + endpoints)
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

    // 4. Both Origin and Destination selected (or updated via search, click, or preset)
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

    // 7. Neither location selected (Reset)
    if (!origin && !destination && !route && !comparison) {
      lastTargetKeyRef.current = null
    }
  }, [map, origin, destination, route, comparison, manualCameraAction])

  return null
}

function MapLocationMarker({ label, location, color, badgeText }) {
  if (!location) {
    return null
  }

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
      text: 'Showing both selected locations',
    }
  }

  if (origin && !destination) {
    return {
      variant: 'origin',
      text: `Focused on Starting Location: ${origin.label || 'Point A'}`,
    }
  }

  if (!origin && destination) {
    return {
      variant: 'destination',
      text: `Focused on Destination: ${destination.label || 'Point B'}`,
    }
  }

  return {
    variant: 'idle',
    text: 'Click map or search to choose locations',
  }
}

function MapView({
  origin,
  destination,
  route,
  comparison,
  electronicCityBounds,
  manualCameraAction,
  onLocationSelect,
  selectionTarget,
}) {
  const shortestCoords = comparison?.shortest_distance?.coordinates?.map(
    ({ latitude, longitude }) => [latitude, longitude],
  )
  const trafficCoords =
    comparison?.traffic_accessibility_aware?.coordinates?.map(
      ({ latitude, longitude }) => [latitude, longitude],
    )

  // Derive leaflet bounds for Electronic City rectangle if provided
  const ecRectBounds = electronicCityBounds
    ? [
        [electronicCityBounds.min_lat, electronicCityBounds.min_lon],
        [electronicCityBounds.max_lat, electronicCityBounds.max_lon],
      ]
    : null

  const statusInfo = getMapStatus(origin, destination, route, comparison)
  const isAlternativeFound = comparison?.status === 'alternative_found'

  return (
    <div className="map-wrapper">
      <MapContainer
        className="route-map"
        center={DEFAULT_MAP_CENTER}
        zoom={13}
        scrollWheelZoom
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapController
          origin={origin}
          destination={destination}
          route={route}
          comparison={comparison}
          manualCameraAction={manualCameraAction}
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
              opacity: 0.85,
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

        {/* Single shortest route fallback when calculate shortest only was used */}
        {!comparison && route && (
          <Polyline
            positions={route.coordinates.map(({ latitude, longitude }) => [
              latitude,
              longitude,
            ])}
            pathOptions={{ color: '#2563EB', weight: 6, opacity: 0.85 }}
          />
        )}

        {/* Snapped endpoints indicators if available */}
        {comparison?.snapped_origin && (
          <CircleMarker
            center={[
              comparison.snapped_origin.latitude,
              comparison.snapped_origin.longitude,
            ]}
            radius={4}
            pathOptions={{ color: '#16a34a', fillColor: '#16a34a', fillOpacity: 0.7 }}
          />
        )}
        {comparison?.snapped_destination && (
          <CircleMarker
            center={[
              comparison.snapped_destination.latitude,
              comparison.snapped_destination.longitude,
            ]}
            radius={4}
            pathOptions={{ color: '#dc2626', fillColor: '#dc2626', fillOpacity: 0.7 }}
          />
        )}

        {/* Origin Marker */}
        <MapLocationMarker
          badgeText="A"
          label={origin?.label || 'Origin'}
          location={origin}
          color="#16a34a"
        />

        {/* Destination Marker */}
        <MapLocationMarker
          badgeText="B"
          label={destination?.label || 'Destination'}
          location={destination}
          color="#dc2626"
        />
      </MapContainer>

      {/* Floating Unobtrusive Map Status Message */}
      <div className="map-status-overlay" aria-live="polite">
        <div className="map-status-chip">
          <span className={`map-status-dot dot-${statusInfo.variant}`} />
          <span className="map-status-text">{statusInfo.text}</span>
        </div>
      </div>

      {/* Floating Map Legend & Selection Target Prompt */}
      <div className="map-floating-overlay" aria-label="Route map legend">
        <div className="map-legend-card">
          <div className="legend-row">
            <span className="legend-swatch shortest-swatch" />
            <span className="legend-text">
              <strong>Shortest</strong> walking path
            </span>
          </div>

          {isAlternativeFound ? (
            <div className="legend-row">
              <span className="legend-swatch traffic-swatch" />
              <span className="legend-text">
                <strong>Traffic-aware</strong> detour
              </span>
            </div>
          ) : comparison ? (
            <div className="legend-row">
              <span className="legend-swatch identical-swatch" />
              <span className="legend-text">
                Traffic route: <em>Identical fallback</em>
              </span>
            </div>
          ) : null}

          {ecRectBounds && (
            <div className="legend-row">
              <span className="legend-swatch ec-swatch" />
              <span className="legend-text">Electronic City dataset zone</span>
            </div>
          )}
        </div>

        <div className="map-prompt-badge">
          <span>Click map to set: </span>
          <strong className={selectionTarget}>
            {selectionTarget === 'origin' ? 'Starting Location (A)' : 'Destination (B)'}
          </strong>
        </div>
      </div>
    </div>
  )
}

export default MapView
