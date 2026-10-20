from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.routing.contracts import Coordinate
from backend.routing.shortest_path import (
    NetworkAreaTooLargeError,
    RouteNetworkUnavailableError,
    RouteNotFoundError,
    ShortestPathService,
)

router = APIRouter(prefix="/api/routes", tags=["routes"])
shortest_path_service = ShortestPathService()


class CoordinateRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90, strict=True)
    longitude: float = Field(ge=-180, le=180, strict=True)


class ShortestRouteRequest(BaseModel):
    origin: CoordinateRequest
    destination: CoordinateRequest


class CoordinateResponse(BaseModel):
    latitude: float
    longitude: float


class ShortestRouteResponse(BaseModel):
    coordinates: list[CoordinateResponse]
    distance_meters: float
    snapped_origin: CoordinateResponse
    snapped_destination: CoordinateResponse


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
