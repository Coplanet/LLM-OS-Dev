from typing import List

from phi.tools import Toolkit
from phi.utils.log import logger


class MyCustomMultipluTool(Toolkit):
    def __init__(self):
        super().__init__(name="my_custom_tools")
        self.register(self.my_custom_multiply)

    def my_custom_multiply(self, args: List[str]) -> str:
        """This function multiplies numbers in a very specific manner.

        :return: The result of the multiplication as string
        """

        # args = [a, b]
        logger.debug("Running Custom Multiply: {}".format("x".join(args)))
        result = 1
        for num in args:
            try:
                result *= float(num)
            except KeyError:
                return "You need to return numbers; {} is not a number!".format(num)
        return "{} = {} [Conf: {}] (from my_custom_multiply)".format(
            "x".join(args), result, result / 10
        )
