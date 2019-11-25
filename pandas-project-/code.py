# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
df=path
bank=pd.read_csv(df)
bank
categorical_var=bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var=bank.select_dtypes(include = 'number')
print(numerical_var)


# code starts here






# code ends here


# --------------
# code starts here


#code ends here
bank=pd.read_csv(path)
bank.drop(['Loan_ID'],inplace=True,axis=1)
banks=bank
print(banks.isnull().sum())
bank_mode=banks.mode()
banks.fillna("bank_mode", inplace = True)
print(banks)



# --------------
# Code starts here
import numpy as np
import pandas as pandas

avg_loan_amount = pd.pivot_table(banks, values='LoanAmount', index=[ 'Gender','Married','Self_Employed'],
                     aggfunc=np.mean)
print(avg_loan_amount)
# code ends here



# --------------
#Create variable 'loan_approved_se' and store the count of results where Self_Employed == Yes and Loan_Status == Y.
loan_approved_se = banks[(banks.Self_Employed == 'Yes') & ( banks.Loan_Status == 'Y')]['Loan_Status'].count()


#Create variable 'loan_approved_nse' and store the count of results where Self_Employed == No and Loan_Status == Y.
loan_approved_nse=banks[(banks.Self_Employed == 'No') & ( banks.Loan_Status == 'Y')]['Loan_Status'].count()

#Loan_Status count is given as 614.
Loan_Status = banks.Loan_Status.count()

#Calculate percentage of loan approval for self employed people and store result in variable
percentage_se =  (loan_approved_se/Loan_Status) *100

#Calculate percentage of loan approval for people who are not self-employed and store the result in variable 'percentage_nse
percentage_nse =   (loan_approved_nse/Loan_Status) *100


# --------------
# code starts here





loan_term=banks['Loan_Amount_Term'].apply(lambda x:x/12)
print(loan_term)

big_loan_term=loan_term.apply(lambda x:x>=25).value_counts().loc[True]
print(big_loan_term)





# code ends here


# --------------
# code starts here

loan_groupby=banks.groupby('Loan_Status')

loan_groupby=loan_groupby['ApplicantIncome', 'Credit_History']

mean_values=loan_groupby.mean()


# code ends here


