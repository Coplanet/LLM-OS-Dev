import json
import uuid
from io import BytesIO
from typing import List, Literal, Optional, Tuple, Union

from PIL import Image
from sqlalchemy import (
    Column,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    delete,
    exc,
    orm,
)

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


class UserBinaryData(Base):
    IMAGE = "image"
    IMAGE_MASK = "image_mask"
    AUDIO = "audio"
    VIDEO = "video"

    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"

    __tablename__ = "user_binary_data"
    session_id = Column(String(50), index=True, nullable=False)

    type = Column(String(20), nullable=False, index=True)
    group_id = Column(String(50), nullable=True, index=True)
    direction = Column(String(20), nullable=False, index=True)

    data = Column(LargeBinary, nullable=False)
    mimetype = Column(String(50), nullable=True, index=True)
    extension = Column(String(10), nullable=True, index=True)

    def data_as_image_thumbnail(
        self, width: int = 256, height: Optional[int] = None
    ) -> bytes:
        output = BytesIO()
        image = Image.open(BytesIO(self.data))
        width = min(width, image.size[0])
        if height is None:
            # compute height based on width and aspect ratio
            height = int(width * image.size[1] / image.size[0])
        height = min(height, image.size[1])
        image.resize((width, height)).save(output, format="webp")
        return output.getvalue()

    @classmethod
    def get_data(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        direction: Literal["upstream", "downstream"],
        **kwargs,
    ) -> orm.query.Query["UserBinaryData"]:
        return (
            db.query(cls)
            .filter_by(session_id=session_id, type=type, direction=direction, **kwargs)
            .order_by(cls.id.desc())
        )

    @classmethod
    def save_data(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        direction: Literal["upstream", "downstream"],
        data: bytes,
        mimetype: str = None,
        extension: str = None,
        auto_commit: bool = True,
    ) -> "UserBinaryData":
        instance = cls(
            session_id=session_id,
            type=type,
            direction=direction,
            data=data,
            mimetype=mimetype,
            extension=extension,
        )
        db.add(instance)
        if auto_commit:
            db.commit()
            db.refresh(instance)
        return instance

    @classmethod
    def save_bulk(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        direction: Literal["upstream", "downstream"],
        data: List[bytes],
        mimetype: Optional[str] = None,
        extension: Optional[str] = None,
        group_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> List["UserBinaryData"]:
        if not group_id:
            group_id = str(uuid.uuid4())

        instances = []

        for data in data:
            instance = cls(
                session_id=session_id,
                type=type,
                direction=direction,
                data=data,
                mimetype=mimetype,
                extension=extension,
                group_id=group_id,
            )
            db.add(instance)
            instances.append(instance)

        if auto_commit:
            db.commit()

            for instance in instances:
                db.refresh(instance)

        return instances

    @classmethod
    def delete_all(cls, db: orm.Session, *args, **kwargs) -> None:
        auto_commit = kwargs.pop("auto_commit", True)
        db.execute(delete(cls).where(*args, **kwargs))
        if auto_commit:
            db.commit()

    @classmethod
    def delete_all_by_type(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        auto_commit: bool = True,
        **kwargs,
    ) -> None:
        kwargs["auto_commit"] = auto_commit
        cls.delete_all(db, cls.session_id == session_id, cls.type == type, **kwargs)

    @classmethod
    def delete_all_by_session_id(
        cls, db: orm.Session, session_id: str, auto_commit: bool = True, **kwargs
    ) -> None:
        kwargs["auto_commit"] = auto_commit
        cls.delete_all(db, cls.session_id == session_id, **kwargs)

    @classmethod
    def delete_all_by_group_id(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        group_id: str,
        auto_commit: bool = True,
        **kwargs,
    ) -> None:
        kwargs["auto_commit"] = auto_commit
        cls.delete_all(
            db,
            cls.session_id == session_id,
            cls.type == type,
            cls.group_id == group_id,
            **kwargs,
        )

    @classmethod
    def get_latest_group(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        direction: Literal["upstream", "downstream"],
    ) -> orm.query.Query["UserBinaryData"]:
        latest_entry = (
            db.query(cls.group_id)
            .filter(
                cls.session_id == session_id,
                cls.type == type,
                cls.direction == direction,
            )
            .order_by(cls.id.desc())
            .first()
        )
        return cls.get_data(
            db, session_id, type, direction=direction, group_id=latest_entry.group_id
        )


class UserNextOp(Base):
    GET_IMAGE_MASK = "get_image_mask"

    __tablename__ = "user_next_ops"
    value_json = None
    session_id = Column(String(50), index=True, nullable=False)
    key = Column(String(255), nullable=False, index=True)
    value = Column(Text, nullable=False, default=lambda: "{}")

    @classmethod
    def get_op(
        cls,
        db: orm.Session,
        session_id: str,
        key: str,
    ) -> Optional["UserNextOp"]:
        obj = (
            db.query(cls)
            .filter_by(session_id=session_id, key=key)
            .order_by(cls.id.desc())
            .first()
        )

        if obj:
            obj.value_json = json.loads(obj.value) if obj.value else {}

        return obj

    @classmethod
    def save_op(
        cls,
        db: orm.Session,
        session_id: str,
        key: str,
        value: Optional[Union[str, dict]] = "{}",
        auto_commit: bool = True,
    ) -> "UserNextOp":
        instance = cls(
            session_id=session_id,
            key=key,
            value=value if isinstance(value, str) else json.dumps(value),
        )
        db.add(instance)
        if auto_commit:
            db.commit()
            db.refresh(instance)
        return instance

    @classmethod
    def delete_all(cls, db: orm.Session, *args, **kwargs) -> None:
        auto_commit = kwargs.pop("auto_commit", True)
        db.execute(delete(cls).where(*args, **kwargs))
        if auto_commit:
            db.commit()

    @classmethod
    def delete_all_by_session_id(
        cls, db: orm.Session, session_id: str, auto_commit: bool = True, **kwargs
    ) -> None:
        kwargs["auto_commit"] = auto_commit
        cls.delete_all(db, cls.session_id == session_id, **kwargs)

    @classmethod
    def delete_all_by_key(
        cls,
        db: orm.Session,
        session_id: str,
        key: str,
        auto_commit: bool = True,
        **kwargs,
    ) -> None:
        kwargs["auto_commit"] = auto_commit
        cls.delete_all(db, cls.session_id == session_id, cls.key == key, **kwargs)
