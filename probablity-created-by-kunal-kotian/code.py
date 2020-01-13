# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# code starts here
df = pd.read_csv(path)
p_a = len(df[df['fico'] > 700])/len(df['fico'])
df1=df[df['purpose']=='debt_consolidation']
p_b=len(df1)/len(df['purpose'])
p_a_b=(p_a and p_b)/p_a
p_a_b
result = (p_a_b == p_a)
print(result)

# code ends here


# --------------
# code starts here
prob_lp = df[df['paid.back.loan'] == 'Yes'].shape[0]/df.shape[0]
prob_cs = df[df['credit.policy'] == 'Yes'].shape[0]/df.shape[0]
new_df = df[df['paid.back.loan'] == 'Yes']

prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0]/new_df.shape[0]
bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
import matplotlib. pyplot as plt
df1=df[df['paid.back.loan']=='No']
df1.plot(kind='bar')
df1.plot.bar()

# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
df['installment'].hist()
plt.show()
df['log.annual.inc'].hist()
plt.show()



# code ends here


