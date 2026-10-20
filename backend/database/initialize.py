from backend.database.session import engine
from backend.observations.models import ObservationRecord


def initialize_database() -> None:
    ObservationRecord.__table__.metadata.create_all(bind=engine)
