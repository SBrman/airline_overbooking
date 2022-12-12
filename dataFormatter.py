import numpy as np
import pandas as pd


# Constants
NORMALIZER = {
    'datedif': {'min': -15, 'max': 1543},
    'arrivehour': {'min': 0.0, 'max': 23.98333333333333}, 
    'departhour': {'min': 0.0, 'max': 23.98333333333333},
    'bookhour': {'min': 0.0, 'max': 23.98333333333333},
    'cost': {'min': 0.0, 'max': 1742298.845146818}, 
    'partysize': {'min': 1.0, 'max': 186.0}
}


ORIGINAL_COLUMNS = ['label', 'partysize', 'overnights', 'datedif', 'arrivehour', 'departhour', 
                    'bookhour', 'type_code', 'cost', 'origin_ANF', 'origin_AQP', 'origin_ARI', 
                    'origin_BBA', 'origin_BOG', 'origin_BRC', 'origin_CCP', 'origin_CJA', 'origin_CJC', 
                    'origin_CLO', 'origin_CNQ', 'origin_COR', 'origin_CPO', 'origin_CRD', 'origin_CUZ', 
                    'origin_EZE', 'origin_FTE', 'origin_GIG', 'origin_IGR', 'origin_IGU', 'origin_IQQ', 
                    'origin_IQT', 'origin_JUJ', 'origin_JUL', 'origin_LIM', 'origin_LSC', 'origin_MDE', 
                    'origin_MDZ', 'origin_MHC', 'origin_MVD', 'origin_NQN', 'origin_PIU', 'origin_PMC', 
                    'origin_PNT', 'origin_POA', 'origin_PSS', 'origin_PUQ', 'origin_ROS', 'origin_SCL', 
                    'origin_SLA', 'origin_TPP', 'origin_TRU', 'origin_TUC', 'origin_TYL', 'origin_USH', 
                    'origin_ZAL', 'origin_ZCO', 'destination_ANF', 'destination_AQP', 'destination_ARI', 
                    'destination_BBA', 'destination_BOG', 'destination_BRC', 'destination_CCP', 'destination_CJA', 
                    'destination_CJC', 'destination_CLO', 'destination_CNQ', 'destination_COR', 'destination_CPO', 
                    'destination_CRD', 'destination_CUZ', 'destination_EZE', 'destination_FTE', 'destination_GIG', 
                    'destination_IGR', 'destination_IGU', 'destination_IQQ', 'destination_IQT', 'destination_JUJ', 
                    'destination_JUL', 'destination_LIM', 'destination_LSC', 'destination_MDE', 'destination_MDZ', 
                    'destination_MHC', 'destination_MVD', 'destination_NQN', 'destination_PIU', 'destination_PMC', 
                    'destination_PNT', 'destination_POA', 'destination_PSS', 'destination_PUQ', 'destination_ROS', 
                    'destination_SCL', 'destination_SLA', 'destination_TPP', 'destination_TRU', 'destination_TUC', 
                    'destination_TYL', 'destination_USH', 'destination_ZAL', 'destination_ZCO', 'departday_Monday', 
                    'departday_Saturday', 'departday_Sunday', 'departday_Thursday', 'departday_Tuesday', 
                    'departday_Wednesday', 'departmonth_2', 'departmonth_3', 'departmonth_4', 'departmonth_5', 
                    'departmonth_6', 'departmonth_7', 'departmonth_8', 'departmonth_9', 'departmonth_10', 
                    'departmonth_11', 'departmonth_12']


class DataFormatter:
    def __init__(self, dataPath):
        self.df = pd.read_csv(dataPath)

    @staticmethod
    def normalize(df, col):
        return (df[col] - NORMALIZER[col]['min']) / (NORMALIZER[col]['max'] - NORMALIZER[col]['min'])

    @staticmethod
    def denormalize(df, col):
        return (df[col] * (NORMALIZER[col]['max'] - NORMALIZER[col]['min'])) + NORMALIZER[col]['min']

    def format_dataset(self, test=False):
        
        df = self.df.astype({'partysize': 'int', 'overnights': 'int', 'datedif': 'int'})

        if test:
            df = df[df.label != 0]
        
        origins_onehot = pd.get_dummies(df.origin, prefix='origin')
        destinations_onehot = pd.get_dummies(df.destination, prefix='destination')
        departday_onehot = pd.get_dummies(df.departday, prefix='departday')
        departmonth_onehot = pd.get_dummies(df.departmonth, prefix='departmonth')
        
        REMOVE = {'origin_AEP', 'destination_AEP', 'departday_Friday', 'departmonth_1'}
        ORIGIN_LIST = [od for od in origins_onehot if od not in REMOVE]
        DESTINATION_LIST = [od for od in destinations_onehot if od not in REMOVE]
        DEPARTMONTH_LIST = [dm for dm in departmonth_onehot if dm not in REMOVE]
        DEPARTDAY_LIST = [dd for dd in departday_onehot if dd not in REMOVE]
        
        df = df.join(origins_onehot[ORIGIN_LIST], how='left')
        df = df.join(destinations_onehot[DESTINATION_LIST], how='left')
        df = df.join(departday_onehot[DEPARTDAY_LIST], how='left')
        df = df.join(departmonth_onehot[DEPARTMONTH_LIST], how='left')
        
        df = df.drop(['origin', 'destination', 'departday', 'departmonth'], axis=1)

        for col in ['datedif', 'arrivehour', 'departhour', 'departhour', 'bookhour', 'cost', 'partysize']:
            df[col] = self.normalize(df, col)

        df = df.replace({'type_code': {1: 0, 2: 1}})
        df['label'] = df['label'].replace({1: 0})
        df['label'] = df['label'].replace({2: 1})

        df = df.astype({'partysize': 'uint8', 'overnights': 'uint8', 'datedif': 'uint16', 'label': 'uint8', 
                        'type_code': 'uint8', 'arrivehour': 'float32', 'departhour': 'float32', 
                        'bookhour': 'float32',  'cost': 'float32',})
        
        for col in ORIGINAL_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df_new = pd.DataFrame()
        df_new = pd.concat([df[col] for col in ORIGINAL_COLUMNS], axis=1)
        
        try:
            df_new = pd.concat([df_new, df['flight']], axis=1)
        except KeyError:
            pass
        
        self.df = df_new
    
    def save_dataset(self, path, test):
        self.format_dataset(test)
        self.df.to_csv(path)
        
       
def main():
    """
    Takes the csv file at source path convert to required format and then save it to the target path.
    """
    source_path = './unorganized/CleanedFullFlights.csv'
    target_path = './data/final_flight_test_cc.csv'
    formatter = DataFormatter(source_path)
    formatter.save_dataset(target_path, test=True)
       
        
if __name__ == '__main__':
    main()
