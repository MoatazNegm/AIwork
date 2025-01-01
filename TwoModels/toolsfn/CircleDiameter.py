def CircleDiameter(radius: str) -> str:
    """
        Description:
            LLMTool that Get the Diameter of a Circle.

        Args:
            radius: The radius of the circle
        Returns:
            The current temperature at the specified location in the specified units, as a string

        return 22.  # A real function should probably actually get the temperature!
      """
    import math
    radiusm = float(radius)
    if radius > 0:
        return str(radiusm * 2)
    else:
        return 'I cannot calculate a diameter of a cirucle whose radius is '+ radiusm