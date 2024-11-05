"""spot_controller controller."""
from spot_driver import SpotDriver


spot = SpotDriver()
spot.move_forward(2)
spot.turn_right(5)
spot.turn_left(5)
spot.move_backward(2)
