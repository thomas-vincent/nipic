import pandas as pd



p_df = (pd.DataFrame([{'descr' : 'T1', 'comp': 'mag', 'exp_size' : 10},
                     {'descr' : 'T1', 'comp': 'ph', 'exp_size' : 10},
                     {'descr' : 'T2', 'comp': None, 'exp_size' : 32}])
        .set_index(['descr', 'comp']))

df = pd.DataFrame([{'PID' : 'CE01', 'descr' : 'T1', 'comp': 'mag', 'size' : 10},
                   {'PID' : 'CE01', 'descr' : 'T2', 'comp': None, 'size' : 7},
                   {'PID' : 'CE01', 'descr' : 'loc', 'comp': None, 'size' : 1}])

df.join(p_df, on=['desc', 'comp'])

print(df)

