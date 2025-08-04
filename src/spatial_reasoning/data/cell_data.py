from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cell:
    id: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def dims(self) -> Tuple[int, int]:
        return self.right - self.left, self.bottom - self.top

    def __str__(self):
        return f"Cell(id={self.id}, left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"

    def to_tuple(self) -> Tuple[int, int, int, int]:
        # x, y, width, height
        return self.left, self.top, self.right - self.left, self.bottom - self.top
