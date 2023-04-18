import numpy as np

def influence_radius(tech_level):
    """
    Gives the radius of influence as a function of the technology level
    of the civilisation.

    TODO: define more rigorously
    """
    return 0.1*np.tan(tech_level * (np.pi / 2))

def inv_influence_radius(inf_radius):
    """
    Gives the tech level corresponding to a given influence radius
    """
    return (2/np.pi) * np.arctan(10*inf_radius)

def sigmoid_growth(time, speed, takeoff_time):
    """
    Gives the current technology level for the agent. Assumes a
    sigmoid-shaped growing curve for the technological capabilities of
    civilisations.

    Arguments can be individual numbers of NumPy arrays of equal length.
    """
    exponent = -speed * (time - takeoff_time)
    return 1/(1+np.exp(exponent))