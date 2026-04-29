from pyMoist.state import MoistState
from pyMoist.convection_tracers import ConvectionTracers

def get_alarm():
    pass

def turn_alarm_off():
    pass

def set_surface_type():
    pass


class Moist:
    def __init__(self):
        pass

    def __call__(self, state: MoistState, convection_tracers: ConvectionTracers):
        # Get alarm - is on (ringing) when pulled
        ALARM_IS_RINGING = get_alarm()

        if ALARM_IS_RINGING:
            ALARM_IS_RINGING = turn_alarm_off()

            set_surface_type()

            # save input winds


