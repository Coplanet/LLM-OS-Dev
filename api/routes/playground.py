from os import getenv

from agno.playground import Playground

from ai.coordinators.generic import get_coordinator as get_generic_leader

######################################################
# Router for the agent playground
######################################################

generic_leader = get_generic_leader()

# Create a playground instance
playground = Playground(agents=[generic_leader])

# Log the playground endpoint with phidata.app
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint("http://localhost:8000")

playground_router = playground.get_router()
