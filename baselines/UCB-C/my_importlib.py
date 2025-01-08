import glob
import importlib


meta_module = {
    # "branin": "Branin",
    # "goldstein": "Goldstein",
    
    'braninhoo': "BraninHoo",
    'braninhoo_constraint': "Braninhoo_Constraint",
    
    'gomez_and_levy': "Gomez_and_Levy",
    'gomez_and_levy_constraint': 'Gomez_and_Levy_Constraint',
    
    'simionescu': "Simionescu",
    'simionescu_constraint': "Simionescu_Constraint",
    
    'hartmann': "Hartmann",
    'hartmann_constraint': "Hartmann_Constraint",
    
    'ackley': "Ackley",
    'ackley_constraint1': "Ackley_Constraint1",
    'ackley_constraint2': "Ackley_Constraint2",
    
    'speed_reducer': "SpeedReducer",
    'speed_reducer_constraint1': 'SpeedReducerConstraint1',
    'speed_reducer_constraint2': 'SpeedReducerConstraint2',
    'speed_reducer_constraint3': 'SpeedReducerConstraint3',
    'speed_reducer_constraint4': 'SpeedReducerConstraint4',
    'speed_reducer_constraint5': 'SpeedReducerConstraint5',
    'speed_reducer_constraint6': 'SpeedReducerConstraint6',
    'speed_reducer_constraint7': 'SpeedReducerConstraint7',
    'speed_reducer_constraint8': 'SpeedReducerConstraint8',
    'speed_reducer_constraint9': 'SpeedReducerConstraint9',
    'speed_reducer_constraint10': 'SpeedReducerConstraint10',
    'speed_reducer_constraint11': 'SpeedReducerConstraint11',
    
    'sensitive_sample': 'SensitiveSample',
    'sensitive_sample_constraint': 'SensitiveSampleConstraint',


    "gas_transmission_2_3_15_obj": "GasTransmission_2_3_15_OBJ",
    "gas_transmission_2_3_15_cst": "GasTransmission_2_3_15_CST",
    

    # "cifar10_cnn_obj": "Cifar10CNN_OBJ",
    # "cifar10_cnn_cst1": "Cifar10CNN_CST1",
    # "cifar10_cnn_cst2": "Cifar10CNN_CST2",
    # "cifar10_cnn_cst3": "Cifar10CNN_CST3",
    # "cifar10_cnn_cst4": "Cifar10CNN_CST4",
    # "cifar10_cnn_cst5": "Cifar10CNN_CST5",
    # "cifar10_cnn_cst6": "Cifar10CNN_CST6",
    # "cifar10_cnn_cst7": "Cifar10CNN_CST7",
    # "cifar10_cnn_cst8": "Cifar10CNN_CST8",
    # "cifar10_cnn_cst9": "Cifar10CNN_CST9",
    # "cifar10_cnn_cst10": "Cifar10CNN_CST10",
    # "two_qubit_obj": "TwoQubit_OBJ",
    # "two_qubit_cst": "TwoQubit_CST",
}


def importfunction(function_module_name):
    module = get_import_path(function_module_name)
    classname = get_function_class_name(function_module_name)
    return getattr(importlib.import_module(module), classname)


def get_import_path(function_module_name):
    return f"func.{function_module_name}"


def get_function_class_name(function_module_name):
    if function_module_name not in meta_module:
        raise Exception(
            f"Function module {function_module_name} not found in meta_module"
        )
    return meta_module[function_module_name]
