import pandas as pd


class GetDummies:
    """
    Since the model can be fed only with the ordered format of the data
    So having the data that the model has already trained with and formatting
    the inference data with that format
    """
    def __init__(self, ref_path):
        self.reference = pd.get_dummies(pd.read_csv(ref_path))
        self.columns = self.reference.columns

    def transform(self, data, desired_columns, change=True):
        data_dummies = pd.get_dummies(pd.DataFrame(data, index=[0]))

        full_data = data_dummies.reindex(columns=self.columns, fill_value=0)
        if change:
            full_data.rename(columns={'VIC_AGE_GROUP__18': 'VIC_AGE_GROUP_18'}, inplace=True)

        print(full_data.columns)
        return full_data[desired_columns]
