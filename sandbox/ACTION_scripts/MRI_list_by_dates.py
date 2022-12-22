import os
import os.path as op
from datetime import datetime

import pandas as pd

mri_root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS'

subjects, session_dates = [], []
for subject_subfolder in os.listdir(mri_root):
    if subject_subfolder.startswith('sub-'):
        subject_folder = op.join(mri_root, subject_subfolder)
        for session_subfolder in os.listdir(subject_folder):
            session_date = datetime.strptime(session_subfolder[4:-2], '%Y%m%d')
            subjects.append(subject_subfolder)
            session_dates.append(session_date.date())
            
report = pd.DataFrame({'Subject' : subjects, 'Acq_Date' : session_dates})
print(report.sort_values(by=['Acq_Date']))
