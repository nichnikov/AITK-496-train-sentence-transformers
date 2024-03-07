import os
from collections import namedtuple
from typing import List
from pydantic_settings import BaseSettings
from pydantic import BaseModel

class Parameters(BaseModel):
    clusters_index: str
    answers_index: str
    stopwords_files: List[str]
    max_hits: int
    chunk_size: int


class TextsDeleteSample(BaseModel):
    """Схема данных для удаления данных по тексту из Индекса"""
    Index: str
    Texts: list[str]
    FieldName: str
    Score: float


ROW = namedtuple("ROW", "SysID, ID, Cluster, ParentModuleID, ParentID, ParentPubList, "
                        "ChildBlockModuleID, ChildBlockID, ModuleID, Topic, Subtopic, DocName, ShortAnswerText")


class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        env_file = os.path.join(os.getcwd(), "data", ".env")
        env_file_encoding = "utf-8"


class ElasticSettings(Settings):
    """Elasticsearch settings."""

    hosts: str
    # index: str
    user_name: str | None
    password: str | None

    max_hits: int = 100
    chunk_size: int = 100

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Returns basic auth tuple if user and password are specified."""
        if self.user_name and self.password:
            return self.user_name, self.password
        return None
