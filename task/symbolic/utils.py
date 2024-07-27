class FlexibleValue:
    def __init__(self, value):
        # Store the value as a string
        self.value = str(value)

    def __eq__(self, other):
        # if the other value is an instance of the FlexibleValue, comparer the value in intern
        if isinstance(other, FlexibleValue):
            return self._normalize(self.value) == self._normalize(other.value)
        # else, compare with the other valuer after normalization
        return self._normalize(self.value) == self._normalize(str(other))

    def __repr(self):
        return f"FlexibleValue({self.value})"

    def _normalize(self, value):
        # delete the signe if it present
        if value.startswith('+'):
            value = value[1:]
        
        # Convert into number if possible for the comparison
        try:
            num_value = float(value)
            if num_value.is_integer():
                return str(int(num_value))
            return str(num_value)
        except ValueError:
            # Return the value not updated if she can not be converted into number
            return value

# defene Point object to format geographic points
class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        # define
        tolerance = 0.3 
        return abs(self.x - other.x) <= tolerance and abs(self.y - other.y) <= tolerance

    def __repr__(self):
        return f"Point({self.x}, {self.y})"