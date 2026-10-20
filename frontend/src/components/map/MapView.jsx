import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip,
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

function MapLocationMarker({ label, location, color }) {
  if (!location) {
    return null
  }

  return (
    <CircleMarker
      center={[location.latitude, location.longitude]}
      radius={9}
      pathOptions={{ color, fillColor: color, fillOpacity: 0.85, weight: 3 }}
    >
      <Tooltip permanent direction="top" offset={[0, -8]}>
        {label}
      </Tooltip>
    </CircleMarker>
  )
}

function MapView({ origin, destination, route, onLocationSelect }) {
  return (
    <MapContainer
      className="route-map"
      center={DEFAULT_MAP_CENTER}
      zoom={5}
      scrollWheelZoom
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <LocationSelector onSelect={onLocationSelect} />
      {route && (
        <Polyline
          positions={route.coordinates.map(({ latitude, longitude }) => [
            latitude,
            longitude,
          ])}
          pathOptions={{ color: '#175cd3', weight: 5 }}
        />
      )}
      <MapLocationMarker label="Origin" location={origin} color="#167647" />
      <MapLocationMarker
        label="Destination"
        location={destination}
        color="#c4320a"
      />
    </MapContainer>
  )
}

export default MapView
