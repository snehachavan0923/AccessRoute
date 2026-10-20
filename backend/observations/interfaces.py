from typing import Protocol, TypeVar

InputT = TypeVar("InputT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class ObservationAnalyzer(Protocol[InputT, ResultT]):
    def analyze(self, source: InputT) -> ResultT: ...
