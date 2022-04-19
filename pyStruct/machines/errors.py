class PyStructException(Exception):
    ...

# ---------------------------------------------
class FileNotFoundException(PyStructException):
    ...

class ModelPathNotFound(FileNotFoundException):
    def __init__(self, message):
        super(ModelPathNotFound, self).__init__(message)



# ---------------------------------------------
class FeatureProcessorException(PyStructException):
    ...


# ---------------------------------------------
class StructureException(PyStructException):
    ...


class LabelNotInStructureTable(StructureException):
    def __init__(self, message):
        super(StructureException, self).__init__(message)

class StructureModelNoneExist(StructureException):
    def __init__(self, message):
        super(StructureModelNoneExist, self).__init__(message)

# ---------------------------------------------
