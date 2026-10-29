from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.accessibility import get_default_accessibility_adapter
from backend.routing.contracts import Coordinate
from backend.routing.shortest_path import (
    NetworkAreaTooLargeError,
    RouteNetworkUnavailableError,
    RouteNotFoundError,
    ShortestPathService,
)
from backend.routing.traffic_aware_path import TrafficRouteComparisonService
from backend.traffic import get_default_vision_adapter

router = APIRouter(prefix="/api/routes", tags=["routes"])
shortest_path_service = ShortestPathService()
comparison_service = TrafficRouteComparisonService()


class CoordinateRequest(BaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class ShortestRouteRequest(BaseModel):
    origin: CoordinateRequest
    destination: CoordinateRequest


class CompareRoutesRequest(BaseModel):
    origin: CoordinateRequest
    destination: CoordinateRequest
    timestamp: str | None = None
    time_index: int | None = None
    traffic_weight: float = Field(default=2.0, ge=0.0, le=50.0)
    accessibility_mode: str = Field(
        default="standard",
        description="Mobility mode: 'standard' walking or 'step_free' (avoids stairs and active barriers)",
    )


class CoordinateResponse(BaseModel):
    latitude: float
    longitude: float


class ShortestRouteResponse(BaseModel):
    coordinates: list[CoordinateResponse]
    distance_meters: float
    snapped_origin: CoordinateResponse
    snapped_destination: CoordinateResponse


class RouteCandidateResponse(BaseModel):
    coordinates: list[CoordinateResponse]
    distance_meters: float
    cost: float
    edges_count: int
    matched_traffic_edges_count: int
    traffic_coverage_percent: float
    average_congestion: float | None = None
    traffic_source: str = "simulated"
    vision_matched_edges_count: int = 0
    simulated_matched_edges_count: int = 0
    steps_count: int = 0
    unpaved_edges_count: int = 0
    obstacles_count: int = 0
    traffic_cost_penalty: float = 0.0
    accessibility_cost_penalty: float = 0.0
    accessibility_constraints_applied: list[str] = []


class RouteComparisonResponse(BaseModel):
    shortest_distance: RouteCandidateResponse
    traffic_accessibility_aware: RouteCandidateResponse
    snapped_origin: CoordinateResponse | None = None
    snapped_destination: CoordinateResponse | None = None
    selected_timestamp: str
    time_index: int
    cost_formula: str
    status: str
    disclaimer: str
    primary_traffic_source: str = "simulated"
    vision_observation_active: bool = False
    accessibility_mode: str = "standard"
    explanation: str = ""



@router.post(
    "/shortest",
    response_model=ShortestRouteResponse,
    responses={
        404: {"description": "No pedestrian path connects the selected locations."},
        422: {"description": "Invalid coordinates or requested area is too large."},
        503: {"description": "Pedestrian network data is unavailable."},
    },
)
def calculate_shortest_route(request: ShortestRouteRequest) -> ShortestRouteResponse:
    origin = Coordinate(request.origin.latitude, request.origin.longitude)
    destination = Coordinate(
        request.destination.latitude,
        request.destination.longitude,
    )

    try:
        route = shortest_path_service.calculate(origin, destination)
    except NetworkAreaTooLargeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RouteNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RouteNetworkUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ShortestRouteResponse(
        coordinates=[
            CoordinateResponse(
                latitude=coordinate.latitude,
                longitude=coordinate.longitude,
            )
            for coordinate in route.coordinates
        ],
        distance_meters=route.distance_meters,
        snapped_origin=CoordinateResponse(
            latitude=route.snapped_origin.latitude,
            longitude=route.snapped_origin.longitude,
        ),
        snapped_destination=CoordinateResponse(
            latitude=route.snapped_destination.latitude,
            longitude=route.snapped_destination.longitude,
        ),
    )


def _to_candidate_response(candidate) -> RouteCandidateResponse:
    return RouteCandidateResponse(
        coordinates=[
            CoordinateResponse(
                latitude=coordinate.latitude,
                longitude=coordinate.longitude,
            )
            for coordinate in candidate.coordinates
        ],
        distance_meters=candidate.distance_meters,
        cost=candidate.cost,
        edges_count=candidate.edges_count,
        matched_traffic_edges_count=candidate.matched_traffic_edges_count,
        traffic_coverage_percent=candidate.traffic_coverage_percent,
        average_congestion=candidate.average_congestion,
        traffic_source=candidate.traffic_source,
        vision_matched_edges_count=candidate.vision_matched_edges_count,
        simulated_matched_edges_count=candidate.simulated_matched_edges_count,
        steps_count=candidate.steps_count,
        unpaved_edges_count=candidate.unpaved_edges_count,
        obstacles_count=candidate.obstacles_count,
        traffic_cost_penalty=candidate.traffic_cost_penalty,
        accessibility_cost_penalty=candidate.accessibility_cost_penalty,
        accessibility_constraints_applied=list(candidate.accessibility_constraints_applied),
    )


@router.post(
    "/compare",
    response_model=RouteComparisonResponse,
    responses={
        404: {"description": "No pedestrian path connects the selected locations."},
        422: {"description": "Invalid coordinates, time index, or area is too large."},
        503: {"description": "Pedestrian network data is unavailable."},
    },
)
def compare_routes(request: CompareRoutesRequest) -> RouteComparisonResponse:
    origin = Coordinate(request.origin.latitude, request.origin.longitude)
    destination = Coordinate(
        request.destination.latitude,
        request.destination.longitude,
    )

    try:
        comparison = comparison_service.compare(
            origin=origin,
            destination=destination,
            timestamp=request.timestamp,
            time_index=request.time_index,
            traffic_weight=request.traffic_weight,
            accessibility_mode=request.accessibility_mode,
        )
    except NetworkAreaTooLargeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RouteNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RouteNetworkUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    snapped_orig = (
        CoordinateResponse(
            latitude=comparison.snapped_origin.latitude,
            longitude=comparison.snapped_origin.longitude,
        )
        if comparison.snapped_origin is not None
        else None
    )
    snapped_dest = (
        CoordinateResponse(
            latitude=comparison.snapped_destination.latitude,
            longitude=comparison.snapped_destination.longitude,
        )
        if comparison.snapped_destination is not None
        else None
    )

    return RouteComparisonResponse(
        shortest_distance=_to_candidate_response(comparison.shortest_distance),
        traffic_accessibility_aware=_to_candidate_response(
            comparison.traffic_accessibility_aware
        ),
        snapped_origin=snapped_orig,
        snapped_destination=snapped_dest,
        selected_timestamp=comparison.selected_timestamp,
        time_index=comparison.time_index,
        cost_formula=comparison.cost_formula,
        status=comparison.status,
        disclaimer=comparison.disclaimer,
        primary_traffic_source=comparison.primary_traffic_source,
        vision_observation_active=comparison.vision_observation_active,
        accessibility_mode=comparison.accessibility_mode,
        explanation=comparison.explanation,
    )


class ObservationStatusResponse(BaseModel):
    vision_registered_count: int
    active_vision_observations_count: int
    active_vision_observations: list[dict[str, Any]]
    accessibility_registered_count: int
    active_accessibility_obstacles_count: int
    active_accessibility_obstacles: list[dict[str, Any]]
    traffic_dataset_fallback_active: bool = True
    traffic_dataset_name: str = "Electronic City, Bengaluru (Simulated)"
    active_routing_traffic_source: str = "simulated"


@router.get("/observation-status", response_model=ObservationStatusResponse)
def get_routing_observation_status() -> ObservationStatusResponse:
    """Return status of live vision observations and accessibility obstacles affecting routing."""
    v_adapter = get_default_vision_adapter()
    a_adapter = get_default_accessibility_adapter()

    active_v = v_adapter.get_active_observations()
    active_a = a_adapter.get_active_obstacles()

    serialized_v = [
        {
            "osm_u": obs.osm_u,
            "osm_v": obs.osm_v,
            "osm_key": obs.osm_key,
            "experimental_congestion_index": obs.experimental_congestion_index,
            "observation_quality_score": obs.observation_quality_score,
            "tracked_vehicles_count": obs.unique_vehicles_observed,
            "is_spatially_calibrated": (
                obs.camera_metadata.is_spatially_calibrated
                if obs.camera_metadata
                else False
            ),
            "camera_id": obs.camera_id,
        }
        for obs in active_v
    ]

    serialized_a = [
        {
            "obstacle_id": obs.obstacle_id,
            "osm_u": obs.osm_u,
            "osm_v": obs.osm_v,
            "osm_key": obs.osm_key,
            "obstacle_type": obs.obstacle_type.value,
            "severity": obs.severity.value,
            "penalty_multiplier": obs.penalty_multiplier,
            "description": obs.description,
            "is_verified": obs.is_verified,
            "confidence": obs.confidence,
        }
        for obs in active_a
    ]

    dataset = comparison_service._traffic_service.dataset
    has_dataset = dataset is not None

    if len(active_v) > 0 and has_dataset:
        active_source = "mixed"
    elif len(active_v) > 0:
        active_source = "vision"
    elif has_dataset:
        active_source = "simulated"
    else:
        active_source = "none"

    return ObservationStatusResponse(
        vision_registered_count=len(v_adapter),
        active_vision_observations_count=len(active_v),
        active_vision_observations=serialized_v,
        accessibility_registered_count=len(active_a),
        active_accessibility_obstacles_count=len(active_a),
        active_accessibility_obstacles=serialized_a,
        traffic_dataset_fallback_active=has_dataset,
        traffic_dataset_name=(
            "Electronic City, Bengaluru (Simulated)" if has_dataset else "None"
        ),
        active_routing_traffic_source=active_source,
    )



class TimePresetResponse(BaseModel):
    id: str
    label: str
    timestamp: str
    time_index: int
    description: str


class TrafficInfoResponse(BaseModel):
    available: bool
    area_name: str
    total_nodes: int
    total_edges: int
    total_timestamps: int
    start_time: str
    end_time: str
    bounds: dict[str, float]
    center: dict[str, float]
    sample_presets: list[TimePresetResponse]
    cost_formula: str
    disclaimer: str


@router.get("/traffic-info", response_model=TrafficInfoResponse)
def get_traffic_info() -> TrafficInfoResponse:
    dataset = comparison_service._traffic_service.dataset
    from backend.routing.traffic_aware_path import (
        COST_FORMULA_DOCUMENTATION,
        SIMULATED_TRAFFIC_DISCLAIMER,
    )

    if dataset is None:
        return TrafficInfoResponse(
            available=False,
            area_name="Electronic City, Bengaluru",
            total_nodes=0,
            total_edges=0,
            total_timestamps=0,
            start_time="",
            end_time="",
            bounds={
                "min_lat": 12.8266519,
                "max_lat": 12.8507217,
                "min_lon": 77.6655778,
                "max_lon": 77.688564,
            },
            center={"latitude": 12.8386868, "longitude": 77.6770709},
            sample_presets=[],
            cost_formula=COST_FORMULA_DOCUMENTATION,
            disclaimer="Traffic dataset is currently not loaded or unavailable on this server.",
        )

    info = dataset.info
    min_lat, max_lat, min_lon, max_lon = (
        getattr(info, "bounds", (12.8266519, 12.8507217, 77.6655778, 77.688564))
    )

    presets = [
        TimePresetResponse(
            id="morning_peak",
            label="Morning Peak (08:30 AM)",
            timestamp="2026-01-01 08:30:00",
            time_index=dataset.get_time_index("2026-01-01 08:30:00")
            if dataset
            else 102,
            description="High commuter volume on arterial avenues; heavy congestion near Toll Gate and tech parks.",
        ),
        TimePresetResponse(
            id="afternoon",
            label="Afternoon Off-Peak (01:00 PM)",
            timestamp="2026-01-01 13:00:00",
            time_index=dataset.get_time_index("2026-01-01 13:00:00")
            if dataset
            else 156,
            description="Moderate vehicular movement; localized congestion near cafeteria zones.",
        ),
        TimePresetResponse(
            id="evening_peak",
            label="Evening Peak (06:30 PM)",
            timestamp="2026-01-01 18:30:00",
            time_index=dataset.get_time_index("2026-01-01 18:30:00")
            if dataset
            else 222,
            description="Severe traffic congestion across outbound Hosur Road corridors and major junctions.",
        ),
        TimePresetResponse(
            id="night_free_flow",
            label="Night / Free-Flow (02:00 AM)",
            timestamp="2026-01-01 02:00:00",
            time_index=dataset.get_time_index("2026-01-01 02:00:00")
            if dataset
            else 24,
            description="Near-zero vehicular density, free-flowing roads with negligible traffic impedance.",
        ),
    ]

    return TrafficInfoResponse(
        available=True,
        area_name=info.area_name,
        total_nodes=info.total_nodes,
        total_edges=info.total_edges,
        total_timestamps=info.total_timestamps,
        start_time=info.start_time,
        end_time=info.end_time,
        bounds={
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        center={
            "latitude": (min_lat + max_lat) / 2,
            "longitude": (min_lon + max_lon) / 2,
        },
        sample_presets=presets,
        cost_formula=COST_FORMULA_DOCUMENTATION,
        disclaimer=SIMULATED_TRAFFIC_DISCLAIMER,
    )


