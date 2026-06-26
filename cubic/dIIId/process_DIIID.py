'''
Process DIIID experimental data to separate out the positive current density region data from the negative current density region.
'''
import pandas as pd

uz_df = pd.read_csv('../../experimental_data/DIIID/DIIID_2005.csv')
uz_df.columns = uz_df.columns.str.strip()
uz_df.sort_values(by='R (m)', inplace=True)

J_positive_df = uz_df[uz_df['J (MA / m^2)'] > 0].copy()
r_positive_df = J_positive_df['R (m)'].copy()

pd.concat([r_positive_df, J_positive_df['J (MA / m^2)']], axis=1).to_csv('../../experimental_data/DIIID/DIIID_2005_positive.csv', index=False)