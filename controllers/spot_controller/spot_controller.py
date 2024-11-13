"""spot_controller controller."""
from spot_driver import SpotDriver

spot = SpotDriver()

while spot.step(spot.get_timestep()) != -1:
    spot.forward(25)
