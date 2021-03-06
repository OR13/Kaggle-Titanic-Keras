
# From Notebook
# https://www.kaggle.com/oriesteele/titanic/been-coding-for-a-week-lol/editnb

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd


def clean(full, titanic):
    # Categorical to Numeric

    def myround(x, base=5):
        return int(base * round(float(x)/base))

    # Transform Sex into binary values 0 and 1
    sex = pd.Series(np.where(full.Sex == 'male', 0, 1), name='Sex')

    # Create a new variable for every unique value of Embarked
    embarked = pd.get_dummies(full.Embarked, prefix='Embarked')
    # print(embarked.head())

    # Create a new variable for every unique value of Embarked
    pclass = pd.get_dummies(full.Pclass, prefix='Pclass')

    # print(pclass.head())

    # Imputation

    # Create dataset
    imputed = pd.DataFrame()

    # Fill missing values of Age with the average of Age (mean)
    imputed['Age'] = full.Age.fillna(full.Age.mean())

    imputed['Age'] = imputed['Age'].map(myround)

    # Fill missing values of Fare with the average of Fare (mean)
    imputed['Fare'] = full.Fare.fillna(full.Fare.mean())
   
    imputed['Fare'] = imputed['Fare'].map(myround)

    # print(imputed['Fare'].head())

    # print(imputed.head())

    # Extract Category from String

    title = pd.DataFrame()
    # we extract the title from each name
    title['Title'] = full['Name'].map(
        lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir":       "Royalty",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "the Countess": "Royalty",
        "Dona":       "Royalty",
        "Mme":        "Mrs",
        "Mlle":       "Miss",
        "Ms":         "Mrs",
        "Mr":        "Mr",
        "Mrs":       "Mrs",
        "Miss":      "Miss",
        "Master":    "Master",
        "Lady":      "Royalty"

    }

    # we map each title
    title['Title'] = title.Title.map(Title_Dictionary)
    title = pd.get_dummies(title.Title)
    #title = pd.concat( [ title , titles_dummies ] , axis = 1 )

    # print(title.head())

    # Extract Category from String

    cabin = pd.DataFrame()

    # replacing missing cabins with U (for Uknown)
    cabin['Cabin'] = full.Cabin.fillna('U')

    # mapping each Cabin value with the cabin letter
    cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin = pd.get_dummies(cabin['Cabin'], prefix='Cabin')

    # print(cabin.head())

    # Extract Category from String

    # a function that extracts each prefix of the ticket, returns 'XXX' if no
    # prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    ticket = pd.DataFrame()

    # Extracting dummy variables from tickets:
    ticket['Ticket'] = full['Ticket'].map(cleanTicket)
    ticket = pd.get_dummies(ticket['Ticket'], prefix='Ticket')

    ticket.shape
    # print(ticket.head())

    # Category from Integer

    family = pd.DataFrame()

    # introducing a new feature : the size of families (including the
    # passenger)
    family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

    # family = pd.get_dummies(family['FamilySize'], prefix='FamilySize')

    # introducing other features based on the family size
    family['Family_Single'] = family['FamilySize'].map(
        lambda s: 1 if s == 1 else 0)
    family['Family_Small'] = family['FamilySize'].map(
        lambda s: 1 if 2 <= s <= 4 else 0)
    family['Family_Large'] = family['FamilySize'].map(
        lambda s: 1 if 5 <= s else 0)

    # print(family.head())

    # Select which features/variables to include in the dataset from the list below:
    # imputed , embarked , pclass , sex , family , cabin , ticket, title

    full_X = pd.concat([imputed , sex ], axis=1)
    # print(full_X.head())

    return full_X


def load():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    full = train.append(test, ignore_index=True)
    titanic = full[:891]

    del train, test

    print('Datasets:', 'full:', full.shape, 'titanic:', titanic.shape)
    
    return full, titanic
