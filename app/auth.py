import hashlib
from time import time
from typing import Optional

import jwt
import streamlit as st

from app.utils import rerun
from helpers.log import logger
from workspace.settings import extra_settings


class User:
    SESSION_KEY: str = "s"
    TIMESTAMP_KEY: str = "t"
    HASH_KEY: str = "h"
    USERNAME_KEY: str = "u"
    AUTH_KEY: str = "auth"

    def __init__(
        self, username: Optional[str] = None, session_id: Optional[str] = None
    ):
        self.username = username
        self.session_id = session_id

    @property
    def is_authenticated(self) -> bool:
        return bool(
            isinstance(self.username, str)
            and isinstance(self.session_id, str)
            and self.username
            and self.session_id
        )

    @property
    def user_id(self) -> str:
        return hashlib.md5(self.username.encode()).hexdigest()

    def __str__(self) -> str:
        return f"User(username={self.username}, session_id={self.session_id})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_auth_param(
        self,
        add_to_query_params: Optional[bool] = False,
    ) -> dict:
        if not self.is_authenticated:
            return {}

        now = time()
        hash = hashlib.sha256(
            f"{now}{extra_settings.secret_key}{self.username}".encode()
        ).hexdigest()
        obj = {
            self.TIMESTAMP_KEY: now,
            self.HASH_KEY: hash,
            self.USERNAME_KEY: self.username,
            self.SESSION_KEY: self.session_id,
        }

        jwt_obj = jwt.encode(obj, extra_settings.secret_key, algorithm="HS256")
        obj = {self.AUTH_KEY: jwt_obj}

        if add_to_query_params:
            st.query_params.update(obj)

        return obj

    @classmethod
    def from_dict(cls, auth: dict) -> "User":
        if (
            auth
            and cls.USERNAME_KEY in auth
            and cls.SESSION_KEY in auth
            and isinstance(auth[cls.USERNAME_KEY], str)
            and isinstance(auth[cls.SESSION_KEY], str)
            and auth[cls.USERNAME_KEY]
            and auth[cls.SESSION_KEY]
        ):
            return cls(
                username=auth[cls.USERNAME_KEY], session_id=auth[cls.SESSION_KEY]
            )
        return cls.guest()

    @classmethod
    def from_jwt(cls, jwt_obj: str) -> "User":
        if not isinstance(jwt_obj, str) or not jwt_obj:
            return cls.guest()

        try:
            return cls.from_dict(
                jwt.decode(jwt_obj, extra_settings.secret_key, algorithms=["HS256"])
            )

        except jwt.InvalidSignatureError:
            st.query_params.pop(cls.AUTH_KEY)
            rerun()

        except Exception as e:
            if cls.AUTH_KEY in st.query_params:
                del st.query_params[cls.AUTH_KEY]
            logger.error(e)
            return cls.guest()

    def get_username(self):
        # Get username from user if not in session state
        if "username" not in st.session_state:
            username_input_container = st.empty()
            username = username_input_container.text_input(
                ":label: Enter your username"
            )
            if username != "":
                st.session_state["username"] = username
                username_input_container.empty()

        # Get username from session state
        username = st.session_state.get("username")  # type: ignore
        return username

    @classmethod
    def load(cls) -> "User":
        if cls.AUTH_KEY in st.query_params:
            return cls.from_jwt(st.query_params[cls.AUTH_KEY])
        return cls.guest()

    @classmethod
    def guest(cls) -> "User":
        return cls()

    @classmethod
    def validate_auth(cls, auth: dict) -> "User":
        if (
            auth
            and isinstance(auth, dict)
            and cls.TIMESTAMP_KEY in auth
            and cls.HASH_KEY in auth
            and cls.USERNAME_KEY in auth
        ):
            timestamp = auth[cls.TIMESTAMP_KEY]
            hash = auth[cls.HASH_KEY]
            username = auth[cls.USERNAME_KEY]
            session_id = auth[cls.SESSION_KEY]
            if (
                session_id
                and timestamp
                and hash
                and username
                and (
                    hash
                    == hashlib.sha256(
                        f"{timestamp}{extra_settings.secret_key}{username}".encode()
                    ).hexdigest()
                )
            ):
                return User(username=username, session_id=session_id)

        return User.guest()


class Auth:
    def get_user(self) -> User:
        return User.load()
