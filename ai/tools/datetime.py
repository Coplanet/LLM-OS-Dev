import json
from datetime import datetime
from typing import Optional

import pytz
from dateutil import tz
from dateutil.relativedelta import relativedelta
from phi.tools import Toolkit

from helpers.log import logger


class DateTimeTools(Toolkit):
    CURRENT_TIMEZONE: str = None  # os.getenv("TZ", None)

    def __init__(
        self,
        current_datetime: Optional[bool] = True,
        relative_datetime: Optional[bool] = True,
        datetime_distance: Optional[bool] = True,
        timezone_conversion: Optional[bool] = True,
        enable_all: bool = False,
    ):
        super().__init__(name="datetime_tools")

        # Register functions in the toolkit
        if current_datetime or enable_all:
            self.register(self.get_current_datetime)
        if relative_datetime or enable_all:
            self.register(self.get_relative_datetime)
        if datetime_distance or enable_all:
            self.register(self.get_datetime_distance)
        if timezone_conversion or enable_all:
            self.register(self.get_timezone_conversion)

        if not self.CURRENT_TIMEZONE:
            self.CURRENT_TIMEZONE = "UTC"

        logger.info(f"Current timezone: {self.CURRENT_TIMEZONE}")

    def _now(self) -> datetime:
        return datetime.now(pytz.timezone(self.CURRENT_TIMEZONE))

    def _date2response(self, operation: str, dt: datetime, timezone: str = None) -> str:
        return json.dumps(
            {
                "operation": operation,
                "result": dt.isoformat(),
                "detail": {
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "hour": dt.hour,
                    "minute": dt.minute,
                    "second": dt.second,
                    "timezone": timezone or self.CURRENT_TIMEZONE,
                },
            }
        )

    def get_current_datetime(self) -> str:
        """Get the current date and time.

        Returns:
            str: JSON string of the current date and time.
        """
        current_dt = self._now()
        logger.info(f"Current date and time is {current_dt}")

        return self._date2response("current_datetime", current_dt)

    def get_relative_datetime(
        self,
        base_datetime: Optional[str] = None,
        delta_years: Optional[int] = 0,
        delta_months: Optional[int] = 0,
        delta_weeks: Optional[int] = 0,
        delta_days: Optional[int] = 0,
        delta_hours: Optional[int] = 0,
        delta_minutes: Optional[int] = 0,
        delta_seconds: Optional[int] = 0,
    ) -> str:
        """Calculate a relative date and time from a base date and time.

        Args:
            base_datetime (str): Base date and time in ISO format (if not provided, the current date and time is used).
            delta_years (int): Number of years to add or subtract.
            delta_months (int): Number of months to add or subtract.
            delta_weeks (int): Number of weeks to add or subtract.
            delta_days (int): Number of days to add or subtract.
            delta_hours (int): Number of hours to add or subtract.
            delta_minutes (int): Number of minutes to add or subtract.
            delta_seconds (int): Number of seconds to add or subtract.

        Returns:
            str: JSON string of the calculated relative date and time.
        """
        try:
            if base_datetime is None:
                base_dt = self._now()
            else:
                base_dt = datetime.fromisoformat(base_datetime)

            # Use relativedelta to accurately support years and months, along with timedelta components.
            relative_dt = base_dt + relativedelta(
                years=delta_years,
                months=delta_months,
                weeks=delta_weeks,
                days=delta_days,
                hours=delta_hours,
                minutes=delta_minutes,
                seconds=delta_seconds,
            )
            logger.info(
                f"Relative date and time from {base_datetime} with delta years {delta_years}, "
                f"months {delta_months}, weeks {delta_weeks}, days {delta_days}, hours {delta_hours}, "
                f"minutes {delta_minutes}, seconds {delta_seconds} is {relative_dt}"
            )

            return self._date2response("relative_datetime", relative_dt)

        except Exception as e:
            logger.error(f"Error calculating relative datetime: {e}")
            return json.dumps({"operation": "relative_datetime", "error": str(e)})

    def get_datetime_distance(self, datetime1: str, datetime2: str) -> str:
        """Calculate the distance between two date and time strings.

        Args:
            datetime1 (str): First date and time in ISO format.
            datetime2 (str): Second date and time in ISO format.

        Returns:
            str: JSON string of the distance in days, hours, and minutes.
        """
        try:
            dt1 = datetime.fromisoformat(datetime1)
            dt2 = datetime.fromisoformat(datetime2)
            delta = abs(dt2 - dt1)
            days, seconds = delta.days, delta.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            logger.info(
                f"Distance between {datetime1} and {datetime2} is {days} days, {hours} hours, {minutes} minutes"
            )
            return json.dumps(
                {
                    "operation": "datetime_distance",
                    "result": {
                        "days": days,
                        "hours": hours,
                        "minutes": minutes,
                        "seconds": seconds,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating datetime distance: {e}")
            return json.dumps({"operation": "datetime_distance", "error": str(e)})

    def get_timezone_conversion(
        self,
        destination_timezone: str,
        base_datetime: Optional[str] = None,
        source_timezone: Optional[str] = None,
    ) -> str:
        """Convert a date and time from a source timezone to a destination timezone.

        If the input datetime is naive (without tzinfo), it is assumed to be in the source timezone.
        If source_timezone is not provided, the server's local timezone is used by default.

        Args:
            destination_timezone (str): The name of the destination timezone (e.g., 'Europe/London').
            base_datetime (str, Optional): Base date and time in ISO format
                                           (if not provided, the current date and time is used).
            source_timezone (str, optional): The name of the source timezone (e.g., 'America/New_York').
                                             If omitted, defaults to the server's local timezone.

        Returns:
            str: JSON string of the converted date and time in ISO format.
        """
        try:
            if base_datetime is None:
                dt = self._now()
            else:
                dt = datetime.fromisoformat(base_datetime)

            # If source_timezone is not provided, default to the server's local timezone.
            if not source_timezone:
                logger.info(
                    "No source timezone provided, defaulting to server timezone"
                )
                src_tz = tz.tzlocal()
            else:
                src_tz = tz.gettz(source_timezone)
                if src_tz is None:
                    raise ValueError(f"Invalid source timezone: {source_timezone}")

            dest_tz = tz.gettz(destination_timezone)
            if dest_tz is None:
                raise ValueError(
                    f"Invalid destination timezone: {destination_timezone}"
                )

            # If datetime is naive, assume it is in the source timezone.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=src_tz)
            converted_dt = dt.astimezone(dest_tz)

            logger.info(
                f"Converted {base_datetime} from timezone {source_timezone or 'server timezone'} to "
                f"{destination_timezone}: {converted_dt.isoformat()}"
            )
            return self._date2response(
                "timezone_conversion", converted_dt, destination_timezone
            )

        except Exception as e:
            logger.error(f"Error converting timezone: {e}")
            return json.dumps({"operation": "timezone_conversion", "error": str(e)})
