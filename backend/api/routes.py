import re
from fastapi import APIRouter, Query
from pydantic import BaseModel
import requests

router = APIRouter()


class GeocodingResult(BaseModel):
    label: str
    address: str
    latitude: float
    longitude: float
    source: str


class GeocodingResponse(BaseModel):
    results: list[GeocodingResult]
    query: str
    status: str


ELECTRONIC_CITY_LANDMARKS = [
    {
        "label": "Electronic City Toll Gate",
        "address": "Hosur Rd, Electronic City Phase 1, Bengaluru",
        "latitude": 12.847657,
        "longitude": 77.671263,
        "aliases": ["toll", "tollgate", "hosur road", "phase 1 entrance"],
    },
    {
        "label": "Infosys Gate 1, Electronic City",
        "address": "Electronics City Phase 1, Hosur Road, Bengaluru",
        "latitude": 12.838520,
        "longitude": 77.665800,
        "aliases": ["infosys", "infy", "gate 1", "infosys campus"],
    },
    {
        "label": "Wipro Technologies Campus",
        "address": "Sarjapur-Electronic City Rd, Phase 1, Bengaluru",
        "latitude": 12.842100,
        "longitude": 77.675000,
        "aliases": ["wipro", "wipro gate", "wipro park"],
    },
    {
        "label": "Velankani Tech Park",
        "address": "Electronics City Phase 1, Bengaluru",
        "latitude": 12.839800,
        "longitude": 77.677500,
        "aliases": ["velankani", "tech park"],
    },
    {
        "label": "Siemens Healthineers",
        "address": "Electronics City Phase 1, Bengaluru",
        "latitude": 12.844000,
        "longitude": 77.674000,
        "aliases": ["siemens"],
    },
    {
        "label": "Electronic City Phase 1 Bus Stop",
        "address": "Near Toll Gate, Electronic City, Bengaluru",
        "latitude": 12.846528,
        "longitude": 77.672245,
        "aliases": ["bus stop", "bmtc", "phase 1 bus"],
    },
    {
        "label": "Hewlett Packard Enterprise (HPE)",
        "address": "Electronics City Phase 1, Bengaluru",
        "latitude": 12.841000,
        "longitude": 77.676500,
        "aliases": ["hp", "hpe", "hewlett packard"],
    },
    {
        "label": "NeoTown Main Entrance",
        "address": "NeoTown Road, Electronic City Phase 1, Bengaluru",
        "latitude": 12.832000,
        "longitude": 77.681000,
        "aliases": ["neotown", "neo town"],
    },
    {
        "label": "Doddathoguru Junction",
        "address": "Doddathoguru, Electronic City Phase 1, Bengaluru",
        "latitude": 12.848000,
        "longitude": 77.683000,
        "aliases": ["doddathoguru", "thoguru"],
    },
    {
        "label": "TCS Electronic City",
        "address": "Think Campus, Electronic City Phase 2, Bengaluru",
        "latitude": 12.848500,
        "longitude": 77.678000,
        "aliases": ["tcs", "tata consultancy"],
    },
    {
        "label": "Electronic City Metro Station (Yellow Line)",
        "address": "Hosur Road, Electronic City, Bengaluru",
        "latitude": 12.846000,
        "longitude": 77.671500,
        "aliases": ["metro", "metro station", "yellow line"],
    },
]


@router.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Accessibility Route Recommendation API"}


@router.get("/health")
def read_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/geocoding/search", response_model=GeocodingResponse)
def search_places(q: str = Query(..., min_length=1, max_length=100)) -> GeocodingResponse:
    query_clean = q.strip().lower()
    results: list[GeocodingResult] = []

    # 1. Match against verified local Electronic City landmarks first
    for landmark in ELECTRONIC_CITY_LANDMARKS:
        label_lower = landmark["label"].lower()
        addr_lower = landmark["address"].lower()
        aliases = landmark.get("aliases", [])

        if (
            query_clean in label_lower
            or query_clean in addr_lower
            or any(query_clean in a or a in query_clean for a in aliases)
        ):
            results.append(
                GeocodingResult(
                    label=landmark["label"],
                    address=landmark["address"],
                    latitude=landmark["latitude"],
                    longitude=landmark["longitude"],
                    source="verified_landmark",
                )
            )

    # 2. If we have at least 3 local matches, we can return immediately
    if len(results) >= 3:
        return GeocodingResponse(
            results=results[:5],
            query=q,
            status="success",
        )

    # 3. Otherwise, query OpenStreetMap Nominatim with safe timeout and User-Agent
    try:
        url = "https://nominatim.openstreetmap.org/search"
        search_query = q
        # If the user didn't mention Bangalore or Bengaluru, bias towards Bangalore
        if not re.search(r"\b(bengaluru|bangalore|electronic city)\b", query_clean):
            search_query = f"{q}, Bengaluru"

        response = requests.get(
            url,
            params={
                "q": search_query,
                "format": "jsonv2",
                "limit": 5,
                "countrycodes": "in",
            },
            headers={
                "User-Agent": "AccessibilityRouteRecommendation/1.0 (academic-research-project)"
            },
            timeout=3.5,
        )

        if response.status_code == 200:
            data = response.json()
            for item in data:
                display_name = item.get("display_name", "")
                parts = [p.strip() for p in display_name.split(",")]
                short_label = parts[0] if parts else "Location"
                address = ", ".join(parts[1:4]) if len(parts) > 1 else display_name

                # Avoid duplicate labels
                if not any(
                    abs(r.latitude - float(item["lat"])) < 0.0001
                    and abs(r.longitude - float(item["lon"])) < 0.0001
                    for r in results
                ):
                    results.append(
                        GeocodingResult(
                            label=short_label,
                            address=address,
                            latitude=float(item["lat"]),
                            longitude=float(item["lon"]),
                            source="openstreetmap",
                        )
                    )
    except Exception:
        # Nominatim unreachable or timed out; fallback safely to local matches
        pass

    return GeocodingResponse(
        results=results[:6],
        query=q,
        status="success" if results else "no_results",
    )
