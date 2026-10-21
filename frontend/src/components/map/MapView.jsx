import { useEffect } from 'react'
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

function MapController({ targetBounds }) {
  const map = useMap()

  useEffect(() => {
    if (targetBounds && targetBounds.length === 2) {
      try {
        map.fitBounds(targetBounds, {
          padding: [45, 45],
          maxZoom: 16,
          animate: true,
        })
      } catch {
        // Safe fallback if bounds are invalid
      }
    }
  }, [map, targetBounds])

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

function MapView({
  origin,
  destination,
  route,
  comparison,
  electronicCityBounds,
  targetBounds,
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

        <MapController targetBounds={targetBounds} />
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
              opacity: 0.8,
            }}
          />
        )}

        {/* Route B: Traffic-aware alternative (GREEN, layered with dashes for overlap distinction) */}
        {comparison && trafficCoords && (
          <Polyline
            positions={trafficCoords}
            pathOptions={{
              color: '#15803D',
              weight: 3.5,
              dashArray: '7, 7',
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
            pathOptions={{ color: '#16a34a', fillColor: '#16a34a', fillOpacity: 0.6 }}
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

      {/* Floating Map Legend & Status Overlay */}
      <div className="map-floating-overlay" aria-label="Route map legend">
        <div className="map-legend-card">
          <div className="legend-row">
            <span className="legend-swatch shortest-swatch" />
            <span className="legend-text">
              <strong>Shortest</strong> walking path
            </span>
          </div>
          <div className="legend-row">
            <span className="legend-swatch traffic-swatch" />
            <span className="legend-text">
              <strong>Traffic-aware</strong> alternative
            </span>
          </div>
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
