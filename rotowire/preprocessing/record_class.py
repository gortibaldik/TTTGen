class Record:
    """ Records holds together tuples (type, entity, value, ha) and allows simple stringification of the tuple"""
    def __init__( self
                , type
                , entity
                , value
                , ha):
        self.type = type
        self.entity = entity
        self.value = value
        self.ha = ha

    def __str__(self):
        return f"{self.value} | {self.entity} | {self.type} | {self.ha}"