import { useEffect, useRef, useState } from 'react'
import MapView from './components/map/MapView'
import { calculateShortestRoute } from './services/routes'

function LocationCoordinates({ label, location }) {
  return (
    <div className="location-coordinates">
      <h2>{label}</h2>
      {location ? (
        <dl>
          <div>
            <dt>Latitude</dt>
            <dd>{location.latitude.toFixed(6)}</dd>
          </div>
          <div>
            <dt>Longitude</dt>
            <dd>{location.longitude.toFixed(6)}</dd>
          </div>
        </dl>
      ) : (
        <p>Not selected</p>
      )}
    </div>
  )
}

function App() {
  const [origin, setOrigin] = useState(null)
  const [destination, setDestination] = useState(null)
  const [selectionTarget, setSelectionTarget] = useState('origin')
  const [route, setRoute] = useState(null)
  const [isCalculating, setIsCalculating] = useState(false)
  const [routeError, setRouteError] = useState('')
  const requestController = useRef(null)

  useEffect(
    () => () => {
      requestController.current?.abort()
    },
    [],
  )

  function clearRouteRequest() {
    requestController.current?.abort()
    requestController.current = null
    setIsCalculating(false)
  }

  function handleLocationSelect(location) {
    clearRouteRequest()
    setRoute(null)
    setRouteError('')

    if (selectionTarget === 'origin') {
      setOrigin(location)
    } else {
      setDestination(location)
    }
  }

  function handleReset() {
    clearRouteRequest()
    setOrigin(null)
    setDestination(null)
    setRoute(null)
    setRouteError('')
  }

  async function handleCalculateRoute() {
    clearRouteRequest()
    const controller = new AbortController()
    requestController.current = controller
    setIsCalculating(true)
    setRoute(null)
    setRouteError('')

    try {
      const result = await calculateShortestRoute(
        origin,
        destination,
        controller.signal,
      )
      if (!controller.signal.aborted) {
        setRoute(result)
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
      }
    }
  }

  return (
    <main className="app">
      <h1>Accessibility Route Map</h1>
      <section aria-label="Select a location">
        <div className="location-controls">
          <fieldset>
            <legend>Choose which point to set on the map</legend>
            <div className="selection-options">
              <label>
                <input
                  type="radio"
                  name="selectionTarget"
                  value="origin"
                  checked={selectionTarget === 'origin'}
                  onChange={() => setSelectionTarget('origin')}
                />
                Origin
              </label>
              <label>
                <input
                  type="radio"
                  name="selectionTarget"
                  value="destination"
                  checked={selectionTarget === 'destination'}
                  onChange={() => setSelectionTarget('destination')}
                />
                Destination
              </label>
            </div>
          </fieldset>
          <button
            type="button"
            className="reset-button"
            onClick={handleReset}
            disabled={!origin && !destination}
          >
            Clear locations
          </button>
        </div>
        <MapView
          origin={origin}
          destination={destination}
          route={route}
          onLocationSelect={handleLocationSelect}
        />
        <p className="map-instructions" aria-live="polite">
          Click the map to set the {selectionTarget}.
        </p>
        <div className="selected-locations" aria-live="polite">
          <LocationCoordinates label="Origin" location={origin} />
          <LocationCoordinates label="Destination" location={destination} />
        </div>
        <div className="route-controls">
          <button
            type="button"
            className="route-button"
            onClick={handleCalculateRoute}
            disabled={!origin || !destination || isCalculating}
          >
            {isCalculating ? 'Calculating shortest route…' : 'Find shortest route'}
          </button>
          {routeError && (
            <p className="route-error" role="alert">
              {routeError}
            </p>
          )}
          {route && (
            <p className="route-distance" aria-live="polite">
              Shortest distance:{' '}
              {route.distance_meters >= 1_000
                ? `${(route.distance_meters / 1_000).toFixed(2)} km`
                : `${route.distance_meters.toFixed(0)} m`}
            </p>
          )}
        </div>
      </section>
    </main>
  )
}

export default App
