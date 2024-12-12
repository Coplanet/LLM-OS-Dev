import json
from typing import Optional, Tuple

from sqlalchemy import Column, String, Text, UniqueConstraint, exc, orm

from .base import Base


class UserConfig(Base):
    __tablename__ = "user_configs"
    value_json = None
    session_id = Column(String(50), index=True, nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False, default=lambda: "{}")

    __table_args__ = (UniqueConstraint("session_id", "key", name="_session_key_uc"),)

    def load_config(self) -> "UserConfig":
        if not isinstance(self.value, str) or not self.value:
            self.value_json = {}
            return self
        try:
            self.value_json = json.loads(self.value)

        except json.JSONDecodeError:
            self.value_json = {}

        if not isinstance(self.value_json, dict):
            self.value_json = {}

        return self

    def __repr__(self):
        return f"{self.session_id} [{self.key}]"

    def save(self, db: orm.Session):
        self.value = json.dumps(self.value_json)
        return super().save(db)

    @classmethod
    def get_models_config(
        cls, db: orm.Session, session_id: str, auto_create: bool = True
    ) -> "UserConfig":
        config, _ = cls.get_config(db, session_id, "models_config", auto_create)
        return config

    @classmethod
    def get_config(
        cls, db: orm.Session, session_id: str, key: str, auto_create: bool = True
    ) -> Tuple[Optional["UserConfig"], bool]:
        try:
            # Try to fetch the object
            instance = db.query(cls).filter_by(session_id=session_id, key=key).first()
            if instance:
                return instance.load_config(), False

            if not auto_create:
                return None, False

            # Create a new instance if it doesn't exist
            instance = cls(
                session_id=session_id, key=key, value=""
            )  # Set default value
            db.add(instance)
            db.commit()
            db.refresh(instance)
            return instance.load_config(), True

        except exc.IntegrityError:
            db.rollback()
            # If there is a race condition, retrieve the existing one after rollback
            instance = db.query(cls).filter_by(session_id=session_id, key=key).first()
            return instance.load_config(), False
