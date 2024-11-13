"""spot_controller controller."""
from spot_driver import SpotDriver
from controller import AnsiCodes

spot = SpotDriver()

while spot.step(spot.get_timestep()) != -1:
