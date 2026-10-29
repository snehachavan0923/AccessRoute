/**
 * Feature Flags Configuration for AccessRoute.
 *
 * For presentation readiness, experimental or unfinished integrations (such as
 * the Traffic & Route Impact tab, which awaits surveyed camera homography calibration)
 * are hidden from the normal presentation UI by default.
 *
 * To restore the Traffic & Route Impact tab:
 *   1. Change ENABLE_TRAFFIC_IMPACT_TAB below to `true`, OR
 *   2. Start Vite with `VITE_ENABLE_TRAFFIC_IMPACT_TAB=true npm run dev`
 */
export const ENABLE_TRAFFIC_IMPACT_TAB =
  import.meta.env.VITE_ENABLE_TRAFFIC_IMPACT_TAB === 'true' || false
