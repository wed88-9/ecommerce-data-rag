from pydantic import BaseModel, Field
from typing import Literal


class Order(BaseModel):
    order_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    product: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    status: Literal["pending", "completed", "cancelled"]
    timestamp: str