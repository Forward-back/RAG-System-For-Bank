from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    enable_web_search: bool = Field(
        default=False,
        description="启用网络搜索作为知识库回退",
    )


class QueryResponse(BaseModel):
    answer: str
