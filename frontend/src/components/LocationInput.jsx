import { useEffect, useRef, useState } from 'react'
import { MapPin, Search, Crosshair, X, Loader2 } from 'lucide-react'
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
  variant = 'origin', // 'origin' (emerald) | 'destination' (rose)
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

  const isOrigin = variant === 'origin'
  const dotColor = isOrigin ? 'bg-emerald-500' : 'bg-rose-500'

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
      className={`relative rounded-xl border p-3 transition-colors ${isActiveTarget
          ? 'border-blue-400 bg-blue-50/40 ring-1 ring-blue-300'
          : 'border-slate-200 bg-white hover:border-slate-300'
        }`}
      ref={containerRef}
    >
      {/* Label and Map-select trigger */}
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <label htmlFor={id} className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
          <span className={`inline-block h-2 w-2 rounded-full ${dotColor}`} />
          <span>{label}</span>
        </label>
        <button
          type="button"
          onClick={onActivateTarget}
          className={`flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition-all ${isActiveTarget
              ? 'bg-blue-600 text-white shadow-xs'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          title="Click on the map to place this point"
        >
          <MapPin className="h-3 w-3" />
          <span>{isActiveTarget ? 'Click map to place' : 'Set from map'}</span>
        </button>
      </div>

      {/* Input field wrapper */}
      <div className="relative flex items-center">
        <div className="pointer-events-none absolute left-2.5 flex items-center text-slate-400">
          <Search className="h-3.5 w-3.5" />
        </div>

        <input
          id={id}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            if (suggestions.length > 0) setIsOpen(true)
          }}
          placeholder={placeholder || 'Search address or enter coordinates...'}
          autoComplete="off"
          className="w-full rounded-lg border border-slate-200 bg-slate-50/70 py-1.5 pr-16 pl-8 text-xs font-medium text-slate-800 placeholder-slate-400 transition-all focus:border-blue-500 focus:bg-white focus:outline-hidden focus:ring-2 focus:ring-blue-100"
        />

        {/* Right side buttons: spinner, clear, GPS */}
        <div className="absolute right-1.5 flex items-center gap-1">
          {isLoading && (
            <Loader2 className="h-3.5 w-3.5 animate-spin-fast text-slate-400" />
          )}

          {query && !isLoading && (
            <button
              type="button"
              onClick={handleClear}
              className="rounded p-0.5 text-slate-400 hover:bg-slate-200 hover:text-slate-600"
              title="Clear input"
              aria-label={`Clear ${label}`}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}

          <button
            type="button"
            onClick={handleUseMyLocation}
            disabled={isLocating}
            className="flex items-center gap-1 rounded bg-slate-100 px-1.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-200 disabled:opacity-50"
            title="Use current location via GPS"
            aria-label={`Use GPS for ${label}`}
          >
            {isLocating ? (
              <Loader2 className="h-3 w-3 animate-spin-fast text-blue-600" />
            ) : (
              <Crosshair className="h-3 w-3 text-slate-600" />
            )}
            <span className="hidden sm:inline">GPS</span>
          </button>
        </div>
      </div>

      {geoError && (
        <p className="mt-1 text-[11px] text-rose-600">{geoError}</p>
      )}

      {value && !isOpen && (
        <p className="mt-1 text-[11px] text-slate-500 font-mono">
          {value.latitude.toFixed(5)}, {value.longitude.toFixed(5)}
        </p>
      )}

      {/* Geocoding Dropdown Suggestions */}
      {isOpen && suggestions.length > 0 && (
        <ul
          className="absolute z-50 left-0 right-0 top-full mt-1 max-h-56 overflow-y-auto rounded-xl border border-slate-200 bg-white p-1 shadow-lg"
          role="listbox"
        >
          {suggestions.map((item, idx) => (
            <li
              key={`${item.latitude}-${item.longitude}-${idx}`}
              onClick={() => handleSelectSuggestion(item)}
              role="option"
              aria-selected="false"
              className="flex cursor-pointer flex-col gap-0.5 rounded-lg px-2.5 py-2 text-xs hover:bg-slate-50 transition-colors"
            >
              <div className="flex items-center justify-between gap-1">
                <span className="font-semibold text-slate-800">{item.label}</span>
                {item.source === 'verified_landmark' && (
                  <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
                    Electronic City
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-500 truncate">{item.address}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
