import logging
import ast

class Config:

    def __init__(self, input_config, config_definition):
        logger = logging.getLogger(__name__)
        
        # Load Configs Here
        self.missing_configs = []
        for config_var in config_definition:
            #check if all expected configs are there
            if config_var[0] not in input_config:
                self.missing_configs.append(config_var[0])
            
            
            else:
                if config_var[0] == 'minObjectsRequired':
                    if isinstance(input_config[config_var[0]],str):
                        
                        listVar = ast.literal_eval(input_config[config_var[0]])

                        logger.debug("The specified value for minObjectsRequired config parameter: %s  was a string. It was parsed into the following list:\n%s\n\n",
                                     input_config[config_var[0]],
                                     listVar)   
                    else:
                        listVar = input_config[config_var[0]]
                        
                    
                    if isinstance(listVar,list):

                        if listVar not in config_var[1]:
                            
                            logger.debug("The specified value for minObjectsRequired config parameter: %s  was not in the list of allowed values:\n %s \n***Algo will early return***\n\n",
                                         listVar,
                                         config_var[1])
                            
                            self.missing_configs.append(config_var[0])
                        
                        else:
                            setattr(self, config_var[0], listVar)
                    
                    else:
                        logger.error('The specified value for minObjectsRequired config parameter: %s is neither a string nor a list. ***Algo will early return***\n\n',
                                             listVar)
                    
                else:
                    try:
                        castedConfigValue = self.getCastedValue(input_config[config_var[0]], config_var)
            
                        if castedConfigValue < config_var[1] or castedConfigValue > config_var[2]:
                            
                            logger.debug("The specified value for %s config parameter: %s  was <  %s  or > %s ***Algo will early return***\n\n",
                                         input_config[config_var[0]],
                                         config_var[0],
                                         config_var[1],
                                         config_var[2])
                            
                            self.missing_configs.append(config_var[0])
                        else:
                            setattr(self, config_var[0], castedConfigValue)
                    
                    except ValueError:
                        self.missing_configs.append(config_var[0])
                        
                        logger.error('The value for this config item could not be casted to the specified type.\nYou provided a value of:%s  for config item%s, which could not be casted to%s\n***Algo will early return***\n\n',
                                     input_config[config_var[0]],
                                     config_var[0],
                                     config_var[len(config_var)-1])
                    



    def is_valid(self):
        return len(self.missing_configs) == 0
    
    
    def getCastedValue(self, configValue, configItemDefinition):
        logger = logging.getLogger(__name__)
        
        specefiedType = configItemDefinition[len(configItemDefinition)-1]
        
        if specefiedType == 'integer':
            return int(configValue)
        elif specefiedType == 'float':
            return float(configValue)
        else:
            logger.error('You specified the type of config item: %s  as: %s\nwe expect either an integer or a float. ***Algo will early return***\n\n',
                                         configItemDefinition[0],
                                         configItemDefinition[len(configItemDefinition)-1])
            
            self.missing_configs.append(configItemDefinition[0])
            
            return configValue