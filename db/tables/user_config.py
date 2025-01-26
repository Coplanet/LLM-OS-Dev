import hashlib
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
    sql,
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
    data_compressed = Column(LargeBinary, nullable=True)

    data_hashsum = Column(String(255), nullable=True, index=True)
    data_compressed_hashsum = Column(String(255), nullable=True, index=True)

    mimetype = Column(String(50), nullable=True, index=True)
    extension = Column(String(10), nullable=True, index=True)

    def data_as_image_thumbnail(
        self,
        height: int = 100,
        width: Optional[int] = None,
        store_compressed: bool = True,
    ) -> bytes:
        output = BytesIO()
        image = Image.open(BytesIO(self.data))
        # compute width based on height and aspect ratio
        if width is None:
            width = int(height * image.size[0] / image.size[1])
        height = min(height, image.size[1])
        width = min(width, image.size[0])
        image.resize((width, height)).save(output, format="webp")
        compressed_data = output.getvalue()
        if store_compressed:
            compressed_data_hashsum = hashlib.sha256(compressed_data).hexdigest()
            self.data_compressed = compressed_data
            self.data_compressed_hashsum = compressed_data_hashsum
        return compressed_data

    @classmethod
    def get_by_id(
        cls, db: orm.Session, session_id: str, id: int
    ) -> Optional["UserBinaryData"]:
        return db.query(cls).filter_by(session_id=session_id, id=id).first()

    @classmethod
    def get_data(
        cls,
        db: orm.Session,
        session_id: str,
        type: Literal["image", "image_mask", "audio", "video"],
        direction: Optional[Literal["upstream", "downstream"]] = None,
        **kwargs,
    ) -> orm.query.Query["UserBinaryData"]:
        if direction is not None:
            kwargs["direction"] = direction
        return (
            db.query(cls)
            .filter_by(session_id=session_id, type=type, **kwargs)
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
        data_hashsum = hashlib.sha256(data).hexdigest()
        instance = cls(
            session_id=session_id,
            type=type,
            direction=direction,
            data=data,
            data_hashsum=data_hashsum,
            mimetype=mimetype,
            extension=extension,
        )
        if type == cls.IMAGE:
            instance.data_as_image_thumbnail()
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
                data_hashsum=hashlib.sha256(data).hexdigest(),
            )
            if type == cls.IMAGE:
                instance.data_as_image_thumbnail()
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
        direction: Optional[Literal["upstream", "downstream"]] = None,
    ) -> orm.query.Query["UserBinaryData"]:
        query = [cls.session_id == session_id, cls.type == type]
        if direction is not None:
            query.append(cls.direction == direction)
        latest_entry = (
            db.query(cls.group_id).filter(*query).order_by(cls.id.desc()).first()
        )
        return (
            cls.get_data(
                db,
                session_id,
                type,
                direction=direction,
                group_id=latest_entry.group_id,
            )
            if latest_entry
            else db.query(UserBinaryData).filter(sql.false())
        )


class UserNextOp(Base):
    AUTH_USER = "auth_user"
    GET_IMAGE_MASK = "get_image_mask"
    EDIT_IMAGE_USING_MASK = "edit_image_using_mask"

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


class UserIntegration(Base):
    __tablename__ = "user_integrations"
    username = Column(String(255), nullable=False, index=True)
    app = Column(String(255), nullable=False, index=True)
    connection_id = Column(String(255), nullable=False, index=True)
    meta = Column(Text, nullable=True, default=lambda: "{}")

    __table_args__ = (UniqueConstraint("username", "app", name="_username_app_uc"),)

    @classmethod
    def get_integrations(
        cls,
        username: str,
        app: Optional[str] = None,
        db: Optional[orm.Session] = None,
    ) -> List["UserIntegration"]:
        def _fetch(db: orm.Session) -> List[UserIntegration]:
            query = db.query(cls).filter_by(username=username)
            if app:
                query = query.filter_by(app=app)
            return query.order_by(cls.id.desc())

        if db:
            return _fetch(db)

        from db.session import get_db_context

        with get_db_context() as db:
            return _fetch(db)

    @classmethod
    def add_integration(
        cls,
        username: str,
        app: str,
        connection_id: str,
        meta: dict = {},
        db: Optional[orm.Session] = None,
        auto_commit: bool = True,
    ) -> Optional["UserIntegration"]:
        try:

            def _add(db_: orm.Session) -> None:
                nonlocal db, auto_commit

                instance = cls(
                    username=username,
                    app=app,
                    connection_id=connection_id,
                    meta=json.dumps(meta),
                )
                db_.add(instance)
                if auto_commit or not db:
                    db_.commit()
                    db_.refresh(instance)

                return instance

            if db:
                return _add(db)

            from db.session import get_db_context

            with get_db_context() as db_:
                return _add(db_)

        except exc.IntegrityError:
            db.rollback()
            return None
