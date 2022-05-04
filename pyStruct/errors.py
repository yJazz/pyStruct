class PyStructException(Exception):
    ...

# ---------------------------------------------
class FileNotFoundException(PyStructException):
    ...

class ModelPathNotFound(FileNotFoundException):
    def __init__(self, message):
        super(ModelPathNotFound, self).__init__(message)

#--------------------- Data structures --------------------------------
class DataError(PyStructException):
    ...

class DataNotExist(DataError):
    def __init__(self, message=''):
        super(DataNotExist, self).__init__(message)



# --------------------- Feature Processor------------------------
class FeatureProcessorException(PyStructException):
    ...



# --------------------- Optimizer ------------------------
class OptimizerException(PyStructException):
    ...

class SampleNotFoundInOptimizationRecord(OptimizerException):
    def __init__(self, message=''):
        super(SampleNotFoundInOptimizationRecord, self).__init__(message)


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
