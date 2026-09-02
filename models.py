"""Shared memory structures the agents pass between each other."""

from typing import List

from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    """One angle of the user's question, plus everything research found for it."""

    question: str
    urls: List[str] = Field(default_factory=list)
    chunks: List[str] = Field(default_factory=list)   # top-k retrieved text chunks
    sources: List[str] = Field(default_factory=list)  # source URL for chunks[i]


class ResearchState(BaseModel):
    """The blackboard. Every agent reads from it and writes back into it."""

    user_query: str
    sub_queries: List[SubQuery] = Field(default_factory=list)
    draft_report: str = ""
    critique: str = ""
    final_report: str = ""
    references: List[str] = Field(default_factory=list)  # index i -> [Source i+1]
    chat_history: List[dict] = Field(default_factory=list)

    def all_urls(self) -> List[str]:
        seen, out = set(), []
        for sq in self.sub_queries:
            for url in sq.urls:
                if url not in seen:
                    seen.add(url)
                    out.append(url)
        return out
