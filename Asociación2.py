#Libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

"""
The csv that is introduced has to have a specific format. The columns will be the different products.
Each row will be each purchase, and True if the product is buyed; false if it's not buyed.

    Pizza   Beer   Napkins   Avocado    Deodorant   ...
 1  True    False   False     False      False
 2  False   True    False     True       True
 3  False   False   True      False      False
 4  True    True    False     False      True
 ...
 
"""
df = pd.read_csv ("association_input.csv")

frequent_itemsets = apriori(df > 0, min_support = 0.06, use_colnames = True)
rules = association_rules (frequent_itemsets, metric = "confidence", min_threshold = 0.8)
rules = association_rules (frequent_itemsets, metric = "lift", min_threshold = 1)
rules.to_csv ("association_output.csv")
