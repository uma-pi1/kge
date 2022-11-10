import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_configspace(hyperparameter_space, seed):
    """
    Reads the config file and produces the necessary variables. Returns a
    configuration space with all variables and their definition.
    :param hyperparameter_space: dictionary containing the variables and their possible
    values as specified in the LibKGE config file
    :param seed: random seed for trial generation
    :return: ConfigurationSpace containing all variables.
    """
    config_space = CS.ConfigurationSpace(seed=seed)

    for p in hyperparameter_space:
        v_name = p["name"]
        v_type = p["type"]

        if v_type == "choice":
            config_space.add_hyperparameter(
                CSH.CategoricalHyperparameter(v_name, choices=p["values"])
            )
        elif v_type == "range":
            log_scale = False
            if "log_scale" in p.keys():
                log_scale = p["log_scale"]
            if type(p["bounds"][0]) is int and type(p["bounds"][1]) is int:
                config_space.add_hyperparameter(
                    CSH.UniformIntegerHyperparameter(
                        name=v_name,
                        lower=p["bounds"][0],
                        upper=p["bounds"][1],
                        default_value=p["bounds"][1],
                        log=log_scale,
                    )
                )
            else:
                config_space.add_hyperparameter(
                    CSH.UniformFloatHyperparameter(
                        name=v_name,
                        lower=p["bounds"][0],
                        upper=p["bounds"][1],
                        default_value=p["bounds"][1],
                        log=log_scale,
                    )
                )
        elif v_type == "fixed":
            config_space.add_hyperparameter(
                CSH.Constant(name=v_name, value=p["value"])
            )
        else:
            raise ValueError("Unknown type {} for hyperparameter variable {}".format(
                v_type, v_name))
    return config_space
