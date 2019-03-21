
# coding: utf-8

# In[ ]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import numpy as np

def extract_all_Error_parameters_from_model(original_y, fittedvalues):
    
    RMSE = np.sqrt(mse(original_y, fittedvalues))
    MAE = mae(original_y, fittedvalues)
    ME = np.mean(original_y - fittedvalues)
    
    from sklearn.metrics import r2_score
    R2 = r2_score(original_y, fittedvalues)
    
    return {'R2': R2, 'RMSE':RMSE, 'MAE':MAE, 'ME':ME}

