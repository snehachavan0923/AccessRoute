import { useEffect, useRef, useState } from 'react'
import { searchPlaces } from '../services/routes'

export default function LocationInput({
  id,
  label,
  value,
  onChange,
  onClear,
  isActiveTarget,
  onActivateTarget,
  placeholder,
}) {
  const [prevValue, setPrevValue] = useState(value)
  const [query, setQuery] = useState(
    value?.label ||
      (value ? `${value.latitude.toFixed(5)}, ${value.longitude.toFixed(5)}` : ''),
  )
  const [suggestions, setSuggestions] = useState([])
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isLocating, setIsLocating] = useState(false)
  const [geoError, setGeoError] = useState('')
  const containerRef = useRef(null)
  const abortControllerRef = useRef(null)

  // Adjust state during render when external prop changes (React recommended pattern)
  if (prevValue !== value) {
    setPrevValue(value)
    setQuery(
      value?.label ||
        (value ? `${value.latitude.toFixed(5)}, ${value.longitude.toFixed(5)}` : ''),
    )
    setSuggestions([])
    setIsOpen(false)
  }

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Debounced geocoding search
  useEffect(() => {
    if (!query || query.length < 2 || (value && query === value.label)) {
      return
    }

    // Check if user entered direct coordinates (e.g. 12.8385, 77.6658)
    const coordMatch = query.match(/^\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)\s*$/)
    let directCoord = null
    if (coordMatch) {
      const lat = parseFloat(coordMatch[1])
      const lon = parseFloat(coordMatch[2])
      if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
        directCoord = {
          label: `Coordinates (${lat.toFixed(5)}, ${lon.toFixed(5)})`,
          address: 'Custom coordinate point',
          latitude: lat,
          longitude: lon,
          source: 'verified_landmark',
        }
      }
    }

    const timeoutId = setTimeout(async () => {
      abortControllerRef.current?.abort()
      const controller = new AbortController()
      abortControllerRef.current = controller
      setIsLoading(true)
      setGeoError('')

      try {
        const data = await searchPlaces(query, controller.signal)
        if (!controller.signal.aborted) {
          const results = data.results || []
          const combined = directCoord ? [directCoord, ...results] : results
          setSuggestions(combined)
          setIsOpen(combined.length > 0)
        }
      } catch {
        if (!controller.signal.aborted) {
          if (directCoord) {
            setSuggestions([directCoord])
            setIsOpen(true)
          } else {
            setGeoError('Search unavailable offline')
            setSuggestions([])
            setIsOpen(false)
          }
        }
      } finally {
        if (abortControllerRef.current === controller) {
          setIsLoading(false)
        }
      }
    }, 350)

    return () => {
      clearTimeout(timeoutId)
      abortControllerRef.current?.abort()
    }
  }, [query, value])

  function handleSelectSuggestion(item) {
    onChange({
      latitude: item.latitude,
      longitude: item.longitude,
      label: item.label,
      address: item.address,
    })
    setQuery(item.label)
    setSuggestions([])
    setIsOpen(false)
  }

  function handleClear() {
    setQuery('')
    setSuggestions([])
    setIsOpen(false)
    setGeoError('')
    onClear()
  }

  function handleUseMyLocation() {
    if (!navigator.geolocation) {
      setGeoError('Geolocation is not supported by your browser')
      return
    }

    setIsLocating(true)
    setGeoError('')

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setIsLocating(false)
        onChange({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          label: 'Current Location',
        })
      },
      (err) => {
        setIsLocating(false)
        if (err.code === 1) {
          setGeoError('Location permission denied')
        } else if (err.code === 2) {
          setGeoError('Location unavailable')
        } else {
          setGeoError('Could not acquire location')
        }
      },
      { timeout: 8000, enableHighAccuracy: true },
    )
  }

  return (
    <div
      className={`location-input-group ${isActiveTarget ? 'active-target' : ''}`}
      ref={containerRef}
    >
      <div className="location-label-row">
        <label htmlFor={id} className="location-input-label">
          {label}
        </label>
        <button
          type="button"
          className={`set-map-chip ${isActiveTarget ? 'selected' : ''}`}
          onClick={onActivateTarget}
          title="Click on the map to set this location"
        >
          {isActiveTarget ? '● Selecting on map' : '○ Click map to set'}
        </button>
      </div>

      <div className="location-field-wrapper">
        <input
          id={id}
          type="text"
          className="location-text-input"
          placeholder={placeholder || 'Search place, landmark, or address...'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            if (suggestions.length > 0) setIsOpen(true)
          }}
          autoComplete="off"
        />

        {isLoading && <span className="input-spinner" aria-hidden="true" />}

        {query && (
          <button
            type="button"
            className="input-clear-btn"
            onClick={handleClear}
            aria-label={`Clear ${label}`}
            title="Clear"
          >
            ✕
          </button>
        )}

        <button
          type="button"
          className="use-gps-btn"
          onClick={handleUseMyLocation}
          disabled={isLocating}
          title="Use current location from browser GPS"
          aria-label={`Use my current location for ${label}`}
        >
          {isLocating ? '…' : '📍 GPS'}
        </button>
      </div>

      {geoError && <p className="field-hint error-hint">{geoError}</p>}

      {value && !isOpen && (
        <p className="coordinate-subtext">
          {value.latitude.toFixed(5)}, {value.longitude.toFixed(5)}
        </p>
      )}

      {isOpen && suggestions.length > 0 && (
        <ul className="suggestions-dropdown" role="listbox">
          {suggestions.map((item, idx) => (
            <li
              key={`${item.latitude}-${item.longitude}-${idx}`}
              className="suggestion-item"
              role="option"
              aria-selected="false"
              onClick={() => handleSelectSuggestion(item)}
            >
              <div className="suggestion-main">
                <span className="suggestion-label">{item.label}</span>
                {item.source === 'verified_landmark' && (
                  <span className="source-pill">Electronic City</span>
                )}
              </div>
              <p className="suggestion-address">{item.address}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
