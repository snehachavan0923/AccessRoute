import { useEffect } from 'react'
import { X, Navigation } from 'lucide-react'

export default function MobileDrawer({ isOpen, onClose, children }) {
  // Close drawer on Escape key
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[2000] lg:hidden">
      {/* Dimmed backdrop */}
      <div
        className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-in drawer container */}
      <aside
        className="fixed inset-y-0 left-0 w-full max-w-sm bg-white shadow-2xl flex flex-col animate-drawer-in"
        role="dialog"
        aria-modal="true"
        aria-label="Route Planning Controls"
      >
        {/* Drawer header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-slate-50/70">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-600 text-white shadow-xs">
              <Navigation className="h-3.5 w-3.5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-800">Plan Walking Route</h2>
              <p className="text-[10px] text-slate-500">AccessRoute Controls</p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-200 hover:text-slate-800 transition-colors"
            aria-label="Close menu"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable controls area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {children}
        </div>
      </aside>
    </div>
  )
}
