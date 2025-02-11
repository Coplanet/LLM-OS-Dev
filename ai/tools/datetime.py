import json
from datetime import datetime, timedelta
from typing import Optional

from phi.tools import Toolkit

from helpers.log import logger


class DateTimeTools(Toolkit):
    def __init__(
        self,
        current_datetime: Optional[bool] = True,
        relative_datetime: Optional[bool] = True,
        datetime_distance: Optional[bool] = True,
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

    def get_current_datetime(self) -> str:
        """Get the current date and time.

        Returns:
            str: JSON string of the current date and time.
        """
        current_dt = datetime.now()
        logger.info(f"Current date and time is {current_dt}")

        return json.dumps(
            {
                "operation": "current_datetime",
                "result": current_dt.isoformat(),
                "detail": {
                    "year": current_dt.year,
                    "month": current_dt.month,
                    "day": current_dt.day,
                    "hour": current_dt.hour,
                    "minute": current_dt.minute,
                    "second": current_dt.second,
                },
            }
        )

    def get_relative_datetime(
        self,
        base_datetime: str,
        delta_weeks: int = 0,
        delta_days: int = 0,
        delta_hours: int = 0,
        delta_minutes: int = 0,
        delta_seconds: int = 0,
    ) -> str:
        """Calculate a relative date and time from a base date and time.

        Args:
            base_datetime (str): Base date and time in ISO format.
            delta_weeks (int): Number of weeks to add or subtract.
            delta_days (int): Number of days to add or subtract.
            delta_hours (int): Number of hours to add or subtract.
            delta_minutes (int): Number of minutes to add or subtract.
            delta_seconds (int): Number of seconds to add or subtract.

        Returns:
            str: JSON string of the calculated relative date and time.
        """
        try:
            base_dt = datetime.fromisoformat(base_datetime)
            relative_dt = base_dt + timedelta(
                weeks=delta_weeks,
                days=delta_days,
                hours=delta_hours,
                minutes=delta_minutes,
                seconds=delta_seconds,
            )
            logger.info(
                f"Relative date and time from {base_datetime} with delta {delta_days} days, "
                f"{delta_hours} hours, {delta_minutes} minutes, {delta_seconds} seconds is {relative_dt}"
            )
            return json.dumps(
                {
                    "operation": "relative_datetime",
                    "result": relative_dt.isoformat(),
                    "detail": {
                        "year": relative_dt.year,
                        "month": relative_dt.month,
                        "day": relative_dt.day,
                        "hour": relative_dt.hour,
                        "minute": relative_dt.minute,
                        "second": relative_dt.second,
                    },
                }
            )

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
