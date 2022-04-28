
def machine_selection(config):
    feature_processor = feature_processor_selector(config.feature_processor)
    optimizer = optimizer_selector(config.optimizer)
    weights_predictor = weights_predictor_selector(config.weights_predictor)
    structure_predictor = structure_predictor_selector(config.structure_predictor)
    reconstructor = reconstructor_selector(config.reconstructor)
    return (feature_processor, optimizer, weights_predictor, structure_predictor, reconstructor)

def feature_processor_selector(feature_processor):
    # Feature Processor 
    if feature_processor.upper() == 'POD_COHERENT_STRENGTH':
        from pyStruct.featureProcessors.podFeatureProcessor import PodCoherentStrength
        return PodCoherentStrength


def optimizer_selector(optimizer):
    if optimizer.upper() == 'POSITIVE_WIEGHTS':
        from pyStruct.optimizers.linearOptm import PositiveWeights
        return PositiveWeights
    elif optimizer.upper() == 'REAL_WEIGHTS':
        from pyStruct.optimizers.linearOptm import RealWeights 
        return RealWeights
    elif optimizer.upper() == 'INTERCEPT_REAL':
        from pyStruct.optimizers.interceptOptm import InterceptRealWeights
        return InterceptRealWeights
    elif optimizer.upper() == 'INTERCEPT_POSITIVE':
        from pyStruct.optimizers.interceptOptm import InterceptPositive
        return InterceptPositive
    else:
        raise NotImplementedError

def weights_predictor_selector(weigths_predictor):
    if weigths_predictor.upper() == 'GB':
        from pyStruct.weightsPredictor.gbRegressor import GbRegressor
        return GbRegressor
    else:
        raise NotImplementedError

def structure_predictor_selector(structure_predictor):
    if structure_predictor.upper() == 'GB':
        from pyStruct.structurePredictor.lookupStructures import GBLookupStructure
        return GBLookupStructure
    else:
        raise NotImplementedError

def reconstructor_selector(reconstructor):
    if reconstructor.upper() == 'LINEAR':
        from pyStruct.reconstructors import LinearReconstruction
        return LinearReconstruction
    elif reconstructor.upper() == 'INTERCEPT':
        from pyStruct.reconstructors import InterceptReconstruction
        return InterceptReconstruction



        