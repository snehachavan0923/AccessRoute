from sqlalchemy.orm import Session

from backend.observations.models import ObservationRecord


class ObservationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, serialized_payload: str) -> ObservationRecord:
        record = ObservationRecord(payload=serialized_payload)
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record
