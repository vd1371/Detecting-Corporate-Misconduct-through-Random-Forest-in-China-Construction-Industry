import logging
import os, sys
import time
import pprint

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score


class Report(object):
    def __init__(self, name = None):
        super(Report, self).__init__()
        
        self.name = name
        
        self.directory = os.path.dirname(__file__) + "/" + self.name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        logging_address = os.path.join(self.directory, 'Report.log')
        self.log = Logger(logger_name = self.name + '-Logger', address = logging_address , mode='a')

    def evaluate_classification(self, y_true, y_pred, inds, label, extra_df = None):
    
        self.log.info(f"----------Classification Report for {label}------------\n" + str(classification_report(y_true, y_pred))+"\n")
        self.log.info(f"----------Confusion Matrix for {label}------------\n" + str(confusion_matrix(y_true, y_pred))+"\n")
        self.log.info(f'----------Accurcay for {label}------------\n'+str(round(accuracy_score(y_true, y_pred),4)))
        
        print (classification_report(y_true, y_pred))
        print (f'Accuracy score for {label}', round(accuracy_score(y_true, y_pred),4))
        print ("------------------------------------------------")
        
        
        report = pd.DataFrame()
        report['Actual'] = y_true
        report['Predicted'] = y_pred
        report['Ind'] = inds
        report.set_index('Ind', inplace=True)
        report.to_csv(self.directory + "/" + f'{label}.csv')
        
        if isinstance(extra_df, pd.DataFrame):
            df = pd.concat([report, extra_df], axis = 1, join = 'inner')
            df.to_csv(self.directory + "/" + f'{label}-ExtraInformation.csv')

    def report_feature_importance(self, best_features, num_top_features, label = "Test" ):
            
        if type(best_features) == dict:
            print ("About to conduct feature importance")
            for k in best_features.keys():
                best_features[k] = abs(best_features[k])
            features_ = pd.Series(OrderedDict(sorted(best_features.items(), key=lambda t: t[1], reverse =True)))
            
            self.log.info(f"Feature importance based on {label}\n" + pprint.pformat(features_.nlargest(num_top_features)))
        
            ax = features_.nlargest(num_top_features).plot(kind='bar', title = label)
            fig = ax.get_figure()
            fig.savefig(self.directory + "/"+ f'{label}-FS.png')
            del fig
            plt.close()
        
        else:
            raise TypeError("--- Incompatibale type of features")

class Logger(object):
    
    instance = None

    def __init__(self, logger_name = 'Logger', address = '',
                 level = logging.DEBUG, console_level = logging.ERROR,
                 file_level = logging.DEBUG, mode = 'w'):
        super(Logger, self).__init__()
        if not Logger.instance:
            logging.basicConfig()
            
            Logger.instance = logging.getLogger(logger_name)
            Logger.instance.setLevel(level)
            Logger.instance.propagate = False
    
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            Logger.instance.addHandler(console_handler)
            
            file_handler = logging.FileHandler(address, mode = mode)
            file_handler.setLevel(file_level)
            formatter = logging.Formatter('%(asctime)s-%(levelname)s- %(message)s')
            file_handler.setFormatter(formatter)
            Logger.instance.addHandler(file_handler)
    
    def _correct_message(self, message):
        output = "\n----------------------------------------------------------\n"
        output += message
        output += "\n---------------------------------------------------------\n"
        return output
        
    def debug(self, message):
        Logger.instance.debug(self._correct_message(message))

    def info(self, message):
        Logger.instance.info(self._correct_message(message))

    def warning(self, message):
        Logger.instance.warning(self._correct_message(message))

    def error(self, message):
        Logger.instance.error(self._correct_message(message))

    def critical(self, message):
        Logger.instance.critical(self._correct_message(message))

    def exception(self, message):
        Logger.instance.exception(self._correct_message(message))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print (f'---- {method.__name__} is about to start ----')
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print (f'---- {method.__name__} is done in {te-ts:.2f} seconds ----')
        return result
    return timed
    

